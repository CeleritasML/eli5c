import logging
import math
import pickle
from time import time

import datasets
import torch
from torch.utils import checkpoint
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

wiki_embedding_path = 'models/wiki40b_index.bin'
bart_model_name = 'facebook/bart-large'
retriever_model_path = 'models/eli5c_retriever_model.bin'
generator_model_path = 'models/eli5c_bart_model_chem1_9.pth'


def load_wiki_passage_and_index():
    wiki40b_snippets = datasets.load_dataset('wiki_snippets', name='wiki40b_en_100_0')['train']
    with open(wiki_embedding_path, 'rb') as f:
        wiki40b_index_flat = pickle.load(f)
    return wiki40b_snippets, wiki40b_index_flat


class ELI5CQAEmbedding(torch.nn.Module):
    def __init__(self, sent_encoder, dim):
        super(ELI5CQAEmbedding, self).__init__()
        self.sent_encoder = sent_encoder
        self.output_dim = 128
        self.project_q = torch.nn.Linear(dim, self.output_dim, bias=False)
        self.project_a = torch.nn.Linear(dim, self.output_dim, bias=False)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def embed_sentences_checkpointed(self, input_ids, attention_mask, checkpoint_batch_size=-1):
        if checkpoint_batch_size < 0 or input_ids.shape[0] < checkpoint_batch_size:
            return self.sent_encoder(input_ids, attention_mask=attention_mask)[1]
        else:
            device = input_ids.device
            input_shape = input_ids.size()
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            head_mask = [None] * self.sent_encoder.config.num_hidden_layers
            extended_attention_mask: torch.Tensor = self.sent_encoder.get_extended_attention_mask(
                attention_mask, input_shape, device
            )
            def partial_encode(*inputs):
                encoder_outputs = self.sent_encoder.encoder(inputs[0], attention_mask=inputs[1], head_mask=head_mask, )
                sequence_output = encoder_outputs[0]
                pooled_output = self.sent_encoder.pooler(sequence_output)
                return pooled_output
            embedding_output = self.sent_encoder.embeddings(
                input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids, inputs_embeds=None
            )
            pooled_output_list = []
            for b in range(math.ceil(input_ids.shape[0] / checkpoint_batch_size)):
                b_embedding_output = embedding_output[b * checkpoint_batch_size: (b + 1) * checkpoint_batch_size]
                b_attention_mask = extended_attention_mask[b * checkpoint_batch_size: (b + 1) * checkpoint_batch_size]
                pooled_output = checkpoint.checkpoint(partial_encode, b_embedding_output, b_attention_mask)
                pooled_output_list.append(pooled_output)
            return torch.cat(pooled_output_list, dim=0)

    def embed_questions(self, q_ids, q_mask, checkpoint_batch_size=-1):
        q_reps = self.embed_sentences_checkpointed(q_ids, q_mask, checkpoint_batch_size)
        return self.project_q(q_reps)

    def embed_answers(self, a_ids, a_mask, checkpoint_batch_size=-1):
        a_reps = self.embed_sentences_checkpointed(a_ids, a_mask, checkpoint_batch_size)
        return self.project_a(a_reps)

    def forward(self, q_ids, q_mask, a_ids, a_mask, checkpoint_batch_size=-1):
        device = q_ids.device
        q_reps = self.embed_questions(q_ids, q_mask, checkpoint_batch_size)
        a_reps = self.embed_answers(a_ids, a_mask, checkpoint_batch_size)
        compare_scores = torch.mm(q_reps, a_reps.t())
        loss_qa = self.ce_loss(compare_scores, torch.arange(compare_scores.shape[1]).to(device))
        loss_aq = self.ce_loss(compare_scores.t(), torch.arange(compare_scores.shape[0]).to(device))
        loss = (loss_qa + loss_aq) / 2
        return loss


def load_retriever_model(device='cuda:0'):
    model_name = 'google/bert_uncased_L-8_H-768_A-12'
    qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name).to(device)
    d_ids = torch.LongTensor(
        [[bert_model.config.bos_token_id if bert_model.config.bos_token_id is not None else 1]]
    ).to(device)
    d_mask = torch.LongTensor([[1]]).to(device)
    sent_dim = bert_model(d_ids, attention_mask=d_mask)[1].shape[-1]
    qa_embedding = ELI5CQAEmbedding(bert_model, sent_dim).to(device)
    param_dict = torch.load(retriever_model_path)
    qa_embedding.load_state_dict(param_dict['model'])
    return qa_tokenizer, qa_embedding


def load_generator_model(device='cuda:0'):
    model_name = bart_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    param_dict = torch.load(generator_model_path)  # has model weights, optimizer, and scheduler states
    model.load_state_dict(param_dict['model'])
    return tokenizer, model


class ELI5cQAModel:
    def __init__(self, device='cpu'):
        self.device = device
        self.wiki_snippets, self.wiki_index = load_wiki_passage_and_index()
        self.retriever_tokenizer, self.retriever = load_retriever_model(device)
        self.generator_tokenizer, self.generator = load_generator_model(device)

    def _question_embed(self, question):
        q_token = self.retriever_tokenizer.batch_encode_plus(question, max_length=128, truncation=True, padding='max_length')
        q_ids, q_mask = (
            torch.LongTensor(q_token['input_ids']).to(self.device),
            torch.LongTensor(q_token['attention_mask']).to(self.device),
        )
        with torch.no_grad():
            q_reps = self.retriever.embed_questions(q_ids, q_mask).cpu().type(torch.float)
        return q_reps.numpy()

    def _query_doc(self, question_embed):
        D, I = self.wiki_index.search(question_embed, 10)
        logging.info('[Support Docs]: %s' % (','.join([str(i) for i in I[0]])))
        res_passages = [self.wiki_snippets[int(i)] for i in I[0]]
        support_doc = "<P> " + " <P> ".join([p["passage_text"] for p in res_passages])
        return support_doc

    def _generate_answer(self, question_and_doc, min_len=64):
        q_token = self.generator_tokenizer.batch_encode_plus([question_and_doc], max_length=512, truncation=True, padding='max_length')
        q_ids, q_mask = (
            torch.LongTensor(q_token['input_ids']).to(self.device),
            torch.LongTensor(q_token['attention_mask']).to(self.device),
        )
        a_token = self.generator_tokenizer.batch_encode_plus(['A'], max_length=360, truncation=True,
                                             padding='max_length')
        a_ids, a_mask = (
            torch.LongTensor(a_token['input_ids']).to(self.device),
            torch.LongTensor(a_token['attention_mask']).to(self.device),
        )
        labels = a_ids[:, 1:].contiguous().clone()
        labels[a_mask[:, 1:].contiguous() == 0] = -100
        model_inputs = {
            'input_ids': q_ids,
            'attention_mask': q_mask,
            'decoder_input_ids': a_ids[:, :-1].contiguous(),
            'labels': labels,
        }
        generated_ids = self.generator.generate(
                input_ids=model_inputs['input_ids'],
                attention_mask=model_inputs['attention_mask'],
                min_length=min_len,
                max_length=128,
                early_stopping=True,
                num_beams=8,
                eos_token_id=self.generator_tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                num_return_sequences=1,
                decoder_start_token_id=self.generator_tokenizer.bos_token_id,
            )[0]
        return self.generator_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def ask(self, question: str, min_len=64):
        q_embed = self._question_embed([question])
        doc = self._query_doc(q_embed)
        question_doc = 'question: {} context: {}'.format(question, doc)
        return self._generate_answer(question_doc, min_len)


if __name__ == '__main__':
    st_time = time()
    model = ELI5cQAModel()
    print('loaded model', time() - st_time)
    answer = model.ask('Why do we, as humans, crave social interaction and attention?')
    print('finish inference', time() - st_time)
    print(answer)
