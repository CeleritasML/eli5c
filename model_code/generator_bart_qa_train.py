import functools
import math
import pickle
from time import time

import datasets
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, get_linear_schedule_with_warmup


class ELI5DatasetS2S(Dataset):
    def __init__(
        self, examples_array, make_doc_fun=None, extra_answer_threshold=3, document_cache=None, training=True
    ):
        self.training = training
        self.data = examples_array
        self.make_doc_function = make_doc_fun
        self.document_cache = {} if document_cache is None else document_cache
        assert not (make_doc_fun is None and document_cache is None)
        # make index of specific question-answer pairs from multi-answers
        if self.training:
            self.qa_id_list = [
                (i, j)
                for i, qa in enumerate(self.data)
                for j, (a, sc) in enumerate(zip(qa['answers']['text'], qa['answers']['score']))
                if j == 0 or sc >= extra_answer_threshold
            ]
        else:
            self.qa_id_list = [(i, 0) for i in range(self.data.num_rows)]

    def __len__(self):
        return len(self.qa_id_list)

    def make_example(self, idx):
        i, j = self.qa_id_list[idx]
        example = self.data[i]
        question = example['title'] + ' ' + example['selftext']
        answer = example['answers']['text'][j]
        q_id = example['q_id']
        if self.make_doc_function is not None:
            self.document_cache[q_id] = self.document_cache.get(q_id, self.make_doc_function(example['title']))
        document = self.document_cache[q_id]
        in_st = 'question: {} context: {}'.format(
            question.lower().replace(' --t--', '').strip(), document.lower().strip(),
        )
        out_st = answer
        return in_st, out_st

    def __getitem__(self, idx):
        return self.make_example(idx)


def make_qa_s2s_model(model_name='facebook/bart-base', from_file=None, device='cuda:0'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if from_file is not None:
        param_dict = torch.load(from_file)  # has model weights, optimizer, and scheduler states
        model.load_state_dict(param_dict['model'])
    return tokenizer, model


def make_qa_s2s_batch(qa_list, tokenizer, max_len=64, max_a_len=360, device='cuda:0'):
    q_ls = [q for q, a in qa_list]
    a_ls = [a for q, a in qa_list]
    q_toks = tokenizer.batch_encode_plus(q_ls, max_length=max_len, truncation=True, padding='max_length')
    q_ids, q_mask = (
        torch.LongTensor(q_toks['input_ids']).to(device),
        torch.LongTensor(q_toks['attention_mask']).to(device),
    )
    a_toks = tokenizer.batch_encode_plus(a_ls, max_length=min(max_len, max_a_len), truncation=True, padding='max_length')
    a_ids, a_mask = (
        torch.LongTensor(a_toks['input_ids']).to(device),
        torch.LongTensor(a_toks['attention_mask']).to(device),
    )
    labels = a_ids[:, 1:].contiguous().clone()
    labels[a_mask[:, 1:].contiguous() == 0] = -100
    model_inputs = {
        'input_ids': q_ids,
        'attention_mask': q_mask,
        'decoder_input_ids': a_ids[:, :-1].contiguous(),
        'labels': labels,
    }
    return model_inputs


def train_qa_s2s_epoch(model, dataset, tokenizer, optimizer, scheduler, args, e=0, curriculum=False):
    model.train()
    # make iterator
    if curriculum:
        train_sampler = SequentialSampler(dataset)
    else:
        train_sampler = RandomSampler(dataset)
    model_collate_fn = functools.partial(
        make_qa_s2s_batch, tokenizer=tokenizer, max_len=args.max_length, device='cuda:0'
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=model_collate_fn)
    epoch_iterator = tqdm(data_loader, desc='Iteration', disable=True)
    # accumulate loss since last print
    loc_steps = 0
    loc_loss = 0.0
    st_time = time()
    for step, batch_inputs in enumerate(epoch_iterator):
        pre_loss = model(**batch_inputs)
        loss = pre_loss[0]
        loss.backward()
        # optimizer
        if step % args.backward_freq == 0:
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


def eval_qa_s2s_epoch(model, dataset, tokenizer, args):
    model.eval()
    # make iterator
    train_sampler = SequentialSampler(dataset)
    model_collate_fn = functools.partial(
        make_qa_s2s_batch, tokenizer=tokenizer, max_len=args.max_length, device='cuda:0'
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=model_collate_fn)
    epoch_iterator = tqdm(data_loader, desc='Iteration', disable=True)
    # accumulate loss since last print
    loc_steps = 0
    loc_loss = 0.0
    st_time = time()
    with torch.no_grad():
        for step, batch_inputs in enumerate(epoch_iterator):
            pre_loss = model(**batch_inputs)
            loss = pre_loss[0]
            loc_loss += loss.item()
            loc_steps += 1
            if step % args.print_freq == 0:
                print(
                    '{:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}'.format(
                        step, len(dataset) // args.batch_size, loc_loss / loc_steps, time() - st_time,
                    )
                )
    print('Total \t L: {:.3f} \t -- {:.3f}'.format(loc_loss / loc_steps, time() - st_time,))


def train_qa_s2s(qa_s2s_model, qa_s2s_tokenizer, s2s_train_set, s2s_valid1_set, s2s_valid2_set, s2s_args):
    s2s_optimizer = AdamW(qa_s2s_model.parameters(), lr=s2s_args.learning_rate, eps=1e-8)
    s2s_scheduler = get_linear_schedule_with_warmup(
        s2s_optimizer,
        num_warmup_steps=400,
        num_training_steps=(s2s_args.num_epochs + 1) * math.ceil(len(s2s_train_set) / s2s_args.batch_size),
    )
    for e in range(s2s_args.next_epoch, s2s_args.next_epoch + s2s_args.num_epochs):
        train_qa_s2s_epoch(
            qa_s2s_model,
            s2s_train_set,
            qa_s2s_tokenizer,
            s2s_optimizer,
            s2s_scheduler,
            s2s_args,
            e,
            curriculum=(e == 0),
        )
        m_save_dict = {
            'model': qa_s2s_model.state_dict(),
            'optimizer': s2s_optimizer.state_dict(),
            'scheduler': s2s_scheduler.state_dict(),
        }
        print('Saving model {}'.format(s2s_args.model_save_name))
        eval_qa_s2s_epoch(qa_s2s_model, s2s_valid1_set, qa_s2s_tokenizer, s2s_args)
        eval_qa_s2s_epoch(qa_s2s_model, s2s_valid2_set, qa_s2s_tokenizer, s2s_args)
        torch.save(m_save_dict, '{}_{}.pth'.format(s2s_args.model_save_name, e))


class ArgumentsS2S:
    def __init__(self):
        self.batch_size = 1
        self.backward_freq = 4
        self.max_length = 512
        self.print_freq = 1000
        self.model_save_name = 'models/eli5c_bart_model'
        self.learning_rate = 2e-4
        self.num_epochs = 3
        self.next_epoch = 0


def load_support_doc(filename):
    f = open(filename, 'rb')
    docs_dat = pickle.load(f)
    q_ids = docs_dat['q_id']
    docs = docs_dat['documents']
    reform_docs = {k: d for k, d in zip(q_ids, docs)}
    f.close()
    return reform_docs


if __name__ == '__main__':
    eli5c = datasets.load_dataset('jsgao/eli5_category')
    eli5c_train_docs = load_support_doc('support_docs/eli5c_train_docs.dat')
    eli5c_val1_docs = load_support_doc('support_docs/eli5c_val1_docs.dat')
    eli5c_val2_docs = load_support_doc('support_docs/eli5c_val2_docs.dat')
    eli5c_train = ELI5DatasetS2S(eli5c['train'], document_cache=eli5c_train_docs)
    eli5c_val1 = ELI5DatasetS2S(eli5c['validation1'], document_cache=eli5c_val1_docs, training=False)
    eli5c_val2 = ELI5DatasetS2S(eli5c['validation2'], document_cache=eli5c_val2_docs, training=False)
    s2s_args = ArgumentsS2S()

    qa_s2s_tokenizer, qa_s2s_model = make_qa_s2s_model(
        model_name='facebook/bart-base',
        from_file=None,
        device='cuda:0'
    )
    train_qa_s2s(qa_s2s_model, qa_s2s_tokenizer, eli5c_train, eli5c_val1, eli5c_val2, s2s_args)
