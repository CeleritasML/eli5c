import pickle
from time import time

import datasets
import numpy as np
import faiss
import pandas as pd
import torch
from tqdm import tqdm

from model_code.retriever_bert_qa_embedding import make_qa_retriever_model


def embed_questions_for_retrieval(q_ls, qa_tokenizer, qa_embedding, device="cuda:0"):
    q_toks = qa_tokenizer.batch_encode_plus(q_ls, max_length=128, truncation=True, padding='max_length')
    q_ids, q_mask = (
        torch.LongTensor(q_toks["input_ids"]).to(device),
        torch.LongTensor(q_toks["attention_mask"]).to(device),
    )
    with torch.no_grad():
        q_reps = qa_embedding.embed_questions(q_ids, q_mask).cpu().type(torch.float)
    return q_reps.numpy()


def query_qa_dense_index(
        questions, qa_embedding, qa_tokenizer, wiki_passages, wiki_index, st_time, n_results=10, device="cuda:0"
):
    q_rep = embed_questions_for_retrieval(questions, qa_tokenizer, qa_embedding, device=device)
    print('Computed embeddings of all %d questions' % len(questions), time() - st_time)
    D, I = wiki_index.search(q_rep, n_results)
    print('Finish ANN search of all %d questions' % len(questions), time() - st_time)
    res_passages_lst = [[wiki_passages[int(i)] for i in i_lst] for i_lst in I]
    support_doc_lst = [
        "<P> " + " <P> ".join([p["passage_text"] for p in res_passages]) for res_passages in res_passages_lst
    ]
    all_res_lists = []
    for (res_passages, dl) in zip(res_passages_lst, D):
        res_list = [dict([(k, p[k]) for k in wiki_passages.column_names]) for p in res_passages]
        for r, sc in zip(res_list, dl):
            r["score"] = float(sc)
        all_res_lists += [res_list[:]]
    return support_doc_lst, all_res_lists


if __name__ == '__main__':
    wiki40b_snippets = datasets.load_dataset('wiki_snippets', name='wiki40b_en_100_0')['train']
    tokenizer, model = make_qa_retriever_model(from_file='models/eli5c_retriever_model_val-1_l-8_h-768_b-512-512_9.pth')
    wiki40b_passage_reps = np.memmap(
        'wiki40b_passages_reps_32_l-8_h-768_b-512-512.dat',
        dtype='float32', mode='r',
        shape=(wiki40b_snippets.num_rows, 128)
    )
    wiki40b_index_flat = faiss.IndexFlatL2(128)
    wiki40b_index_flat.add(wiki40b_passage_reps)
    st_time = time()

    def query_all(q_set, filename):
        support_doc, dense_res_list = query_qa_dense_index(
            [q['title'] for q in q_set],
            model,
            tokenizer,
            wiki40b_snippets,
            wiki40b_index_flat,
            st_time=st_time,
            n_results=10
        )
        docs = {
            'q_id': [q['q_id'] for q in q_set],
            'documents': support_doc,
            'doc_res_list': dense_res_list
        }
        f = open(filename, 'wb')
        pickle.dump(docs, f)
        f.close()
        print('Save to %s' % filename, time() - st_time)

    eli5c = datasets.load_dataset('jsgao/eli5_category')
    query_all(eli5c['train'], 'eli5c_train_docs.dat')
    query_all(eli5c['validation1'], 'eli5c_val1_docs.dat')
    query_all(eli5c['validation2'], 'eli5c_val2_docs.dat')
