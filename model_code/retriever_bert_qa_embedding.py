import math
from time import time

import datasets
import numpy as np
import torch

from model_code.retriever_bert_qa_train import make_qa_retriever_model


def embed_questions_for_retrieval(q_ls, qa_tokenizer, qa_embedding, max_length=128, device="cuda:0"):
    q_toks = qa_tokenizer.batch_encode_plus(q_ls, max_length=max_length, truncation=True, padding='max_length')
    q_ids, q_mask = (
        torch.LongTensor(q_toks["input_ids"]).to(device),
        torch.LongTensor(q_toks["attention_mask"]).to(device),
    )
    with torch.no_grad():
        q_reps = qa_embedding.embed_questions(q_ids, q_mask).cpu().type(torch.float)
    return q_reps.numpy()


def make_qa_dense_index(
    qa_embedding,
    qa_tokenizer,
    qustions_set,
    batch_size=512,
    max_length=128,
    index_name="kilt_passages_reps.dat",
    dtype="float32",
    device="cuda:0",
):
    st_time = time()
    fp = np.memmap(index_name, dtype=dtype, mode="w+", shape=(qustions_set.num_rows, 128))
    n_batches = math.ceil(qustions_set.num_rows / batch_size)
    for i in range(n_batches):
        questions = [q for q in qustions_set[i * batch_size: (i + 1) * batch_size]['title']]
        reps = embed_questions_for_retrieval(questions, qa_tokenizer, qa_embedding, max_length, device)
        fp[i * batch_size: (i + 1) * batch_size] = reps
        if i % 2 == 0:
            print(i, time() - st_time)


if __name__ == '__main__':
    tokenizer, model = make_qa_retriever_model(from_file='models/eli5c_retriever_model_val-1_l-8_h-768_b-512-512_9.pth')
    eli5c = datasets.load_dataset('jsgao/eli5_category')
    make_qa_dense_index(model, tokenizer, eli5c['train'], device='cuda:0', index_name='embeds/eli5c_train.dat')
    make_qa_dense_index(model, tokenizer, eli5c['validation1'], device='cuda:0', index_name='embeds/eli5c_val1.dat')
    make_qa_dense_index(model, tokenizer, eli5c['validation2'], device='cuda:0', index_name='embeds/eli5c_val2.dat')
