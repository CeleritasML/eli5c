import functools
import math
from random import choice, randint
from time import time

import datasets
import torch
from torch.utils import checkpoint
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup


class ELI5CQARetriever(Dataset):
    def __init__(self, examples_array, min_answer_length=64, max_answer_length=512, training=True, n_samples=None):
        self.data = examples_array
        self.min_length = min_answer_length
        self.max_length = max_answer_length
        self.training = training
        self.n_samples = self.data.num_rows if n_samples is None else n_samples

    def __len__(self):
        return self.n_samples

    def make_example(self, idx):
        example = self.data[idx]
        question = example['title']
        if self.training:
            # if training, random choose 1 answer, random slicing with length in (self.min_length, self.max_length)
            answers = [a for i, (a, sc) in enumerate(zip(example['answers']['text'], example['answers']['score']))]
            answer_tab = choice(answers).split(' ')
            start_idx = randint(0, max(0, len(answer_tab) - self.min_length))
            end_idx = min(len(answer_tab), start_idx + self.max_length)
            answer_span = ' '.join(answer_tab[start_idx:end_idx])
        else:
            # if validation, use the best answer
            answer_span = example['answers']['text'][0]
        return question, answer_span

    def __getitem__(self, idx):
        return self.make_example(idx % self.data.num_rows)


class ELI5CQAEmbedding(torch.nn.Module):
    def __init__(self, sent_encoder, dim):
        super(ELI5CQAEmbedding, self).__init__()
        self.sent_encoder = sent_encoder
        self.output_dim = 128
        self.project_q = torch.nn.Linear(dim, self.output_dim, bias=False)
        self.project_a = torch.nn.Linear(dim, self.output_dim, bias=False)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def embed_sentences_checkpointed(self, input_ids, attention_mask, checkpoint_batch_size=-1):
        # reproduces BERT forward pass with checkpointing
        if checkpoint_batch_size < 0 or input_ids.shape[0] < checkpoint_batch_size:
            return self.sent_encoder(input_ids, attention_mask=attention_mask)[1]
        else:
            # prepare implicit variables
            device = input_ids.device
            input_shape = input_ids.size()
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            head_mask = [None] * self.sent_encoder.config.num_hidden_layers
            extended_attention_mask: torch.Tensor = self.sent_encoder.get_extended_attention_mask(
                attention_mask, input_shape, device
            )

            # define function for checkpointing
            def partial_encode(*inputs):
                encoder_outputs = self.sent_encoder.encoder(inputs[0], attention_mask=inputs[1], head_mask=head_mask, )
                sequence_output = encoder_outputs[0]
                pooled_output = self.sent_encoder.pooler(sequence_output)
                return pooled_output

            # run embedding layer on everything at once
            embedding_output = self.sent_encoder.embeddings(
                input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids, inputs_embeds=None
            )
            # run encoding and pooling on one mini-batch at a time
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


def make_qa_retriever_model(model_name='google/bert_uncased_L-8_H-768_A-12', from_file=None, device='cuda:0'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name).to(device)
    # run bert_model on a dummy batch to get output dimension
    d_ids = torch.LongTensor(
        [[bert_model.config.bos_token_id if bert_model.config.bos_token_id is not None else 1]]
    ).to(device)
    d_mask = torch.LongTensor([[1]]).to(device)
    sent_dim = bert_model(d_ids, attention_mask=d_mask)[1].shape[-1]
    qa_embedding = ELI5CQAEmbedding(bert_model, sent_dim).to(device)
    if from_file is not None:
        param_dict = torch.load(from_file)  # has model weights, optimizer, and scheduler states
        qa_embedding.load_state_dict(param_dict['model'])
    return tokenizer, qa_embedding


def make_qa_retriever_batch(qa_list, tokenizer, max_len=64, device='cuda:0'):
    q_ls = [q for q, a in qa_list]
    a_ls = [a for q, a in qa_list]
    q_toks = tokenizer.batch_encode_plus(q_ls, max_length=max_len, truncation=True, padding='max_length')
    q_ids, q_mask = (
        torch.LongTensor(q_toks['input_ids']).to(device),
        torch.LongTensor(q_toks['attention_mask']).to(device),
    )
    a_toks = tokenizer.batch_encode_plus(a_ls, max_length=max_len, truncation=True, padding='max_length')
    a_ids, a_mask = (
        torch.LongTensor(a_toks['input_ids']).to(device),
        torch.LongTensor(a_toks['attention_mask']).to(device),
    )
    return q_ids, q_mask, a_ids, a_mask


def train_qa_retriever_epoch(model, dataset, tokenizer, optimizer, scheduler, args, e=0):
    model.train()
    # make iterator
    train_sampler = RandomSampler(dataset)
    model_collate_fn = functools.partial(
        make_qa_retriever_batch, tokenizer=tokenizer, max_len=args.max_length, device='cuda:0'
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=model_collate_fn)
    epoch_iterator = tqdm(data_loader, desc='Iteration', disable=True)
    # accumulate loss since last print
    loc_steps = 0
    loc_loss = 0.0
    st_time = time()
    for step, batch in enumerate(epoch_iterator):
        q_ids, q_mask, a_ids, a_mask = batch
        pre_loss = model(q_ids, q_mask, a_ids, a_mask, checkpoint_batch_size=args.checkpoint_batch_size)
        loss = pre_loss.sum()
        # optimizer
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        # some printing within the epoch
        loc_loss += loss.item()
        loc_steps += 1
        if step % args.print_freq == 0 or step == 1:
            print(
                '{:2d} {:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}'.format(
                    e, step, len(dataset) // args.batch_size, loc_loss / loc_steps, time() - st_time,
                )
            )
            loc_loss = 0
            loc_steps = 0


def evaluate_qa_retriever(model, dataset, tokenizer, args):
    model.eval()
    # make iterator
    eval_sampler = SequentialSampler(dataset)
    model_collate_fn = functools.partial(
        make_qa_retriever_batch, tokenizer=tokenizer, max_len=args.max_length, device='cuda:0'
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=eval_sampler, collate_fn=model_collate_fn)
    epoch_iterator = tqdm(data_loader, desc='Iteration', disable=True)
    tot_loss = 0.0
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            q_ids, q_mask, a_ids, a_mask = batch
            loss = model(q_ids, q_mask, a_ids, a_mask)
            tot_loss += loss.item()
        return tot_loss / (step + 1)


def train_qa_retriever(model, tokenizer, train_set, valid_set1, valid_set2, model_args):
    qar_optimizer = AdamW(model.parameters(), lr=model_args.learning_rate, eps=1e-8)
    qar_scheduler = get_linear_schedule_with_warmup(
        qar_optimizer,
        num_warmup_steps=100,
        num_training_steps=(model_args.num_epochs + 1) * math.ceil(len(train_set) / model_args.batch_size),
    )
    for e in range(model_args.num_epochs):
        train_qa_retriever_epoch(model, train_set, tokenizer, qar_optimizer, qar_scheduler, model_args, e)
        m_save_dict = {
            'model': model.state_dict(),
            'optimizer': qar_optimizer.state_dict(),
            'scheduler': qar_scheduler.state_dict(),
        }
        print('Saving model {}'.format(model_args.model_save_name))
        torch.save(m_save_dict, '{}_{}.pth'.format(model_args.model_save_name, e))
        eval_loss1 = evaluate_qa_retriever(model, valid_set1, tokenizer, model_args)
        eval_loss2 = evaluate_qa_retriever(model, valid_set2, tokenizer, model_args)
        print('Evaluation 1 loss epoch {:4d}: {:.3f}'.format(e, eval_loss1))
        print('Evaluation 2 loss epoch {:4d}: {:.3f}'.format(e, eval_loss2))


class ArgumentsQAR:
    def __init__(self):
        self.batch_size = 512
        self.max_length = 128
        self.checkpoint_batch_size = 16
        self.print_freq = 1
        self.pretrained_model_name = 'google/bert_uncased_L-8_H-768_A-12'
        self.model_save_name = 'models/eli5c_retriever_model'
        self.learning_rate = 2e-4
        self.num_epochs = 10


if __name__ == '__main__':
    eli5c = datasets.load_dataset('jsgao/eli5_category')
    eli5c_train_set = ELI5CQARetriever(eli5c['train'])
    eli5c_val_set1 = ELI5CQARetriever(eli5c['validation1'])
    eli5c_val_set2 = ELI5CQARetriever(eli5c['validation2'])
    qar_args = ArgumentsQAR()
    qar_tokenizer, qar_model = make_qa_retriever_model(
        model_name=qar_args.pretrained_model_name,
        from_file=None,
        device='cuda:0'
    )
    train_qa_retriever(qar_model, qar_tokenizer, eli5c_train_set, eli5c_val_set1, eli5c_val_set2, qar_args)
