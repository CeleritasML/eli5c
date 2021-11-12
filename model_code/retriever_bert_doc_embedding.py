import math
from time import time

import datasets
import numpy as np
import torch

from model_code.retriever_bert_qa_embedding import make_qa_retriever_model


def embed_passages_for_retrieval(passages, qa_tokenizer, qa_embedding, max_length=128, device='cuda:0'):
    a_toks = qa_tokenizer.batch_encode_plus(passages, max_length=max_length, truncation=True, padding='max_length')
    a_ids, a_mask = (
        torch.LongTensor(a_toks['input_ids']).to(device),
        torch.LongTensor(a_toks['attention_mask']).to(device),
    )
    with torch.no_grad():
        a_reps = qa_embedding.embed_answers(a_ids, a_mask).cpu().type(torch.float)
    return a_reps.numpy()


def make_qa_dense_index(
    qa_embedding,
    qa_tokenizer,
    passages_set,
    batch_size=512,
    max_length=128,
    index_name='kilt_passages_reps.dat',
    dtype='float32',
    device='cuda:0',
):
    st_time = time()
    fp = np.memmap(index_name, dtype=dtype, mode='w+', shape=(passages_set.num_rows, 128))
    n_batches = math.ceil(passages_set.num_rows / batch_size)
    for i in range(n_batches):
        passages = [p for p in passages_set[i * batch_size: (i + 1) * batch_size]['passage_text']]
        reps = embed_passages_for_retrieval(passages, qa_tokenizer, qa_embedding, max_length, device)
        fp[i * batch_size: (i + 1) * batch_size] = reps
        if i % 50 == 0:
            print(i, time() - st_time)


if __name__ == '__main__':
    wiki40b_snippets = datasets.load_dataset('wiki_snippets', name='wiki40b_en_100_0')['train']
    tokenizer, model = make_qa_retriever_model(from_file='models/eli5c_retriever_model_9.pth')
    make_qa_dense_index(
        model, tokenizer, wiki40b_snippets, device='cuda:0',
        index_name='embeds/wiki40b.dat'
    )
