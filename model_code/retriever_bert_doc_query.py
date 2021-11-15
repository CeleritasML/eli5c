import pickle
from time import time

import datasets
import numpy as np
import faiss


def query_qa_dense_index(questions_vectors, wiki_passages, wiki_index, st_time, n_results=10):
    print('Start query top %d support vector for %d questions' % (n_results, len(questions_vectors)), time() - st_time)
    D, I = wiki_index.search(questions_vectors, n_results)
    print('Finish query top %d support vector for %d questions' % (n_results, len(questions_vectors)), time() - st_time)
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
    st_time = time()
    wiki40b_snippets = datasets.load_dataset('wiki_snippets', name='wiki40b_en_100_0')['train']
    wiki40b_file_name = 'embeds/wiki40b.dat'
    wiki40b_passage_reps = np.memmap(wiki40b_file_name, dtype='float32',
                                     mode='r', shape=(wiki40b_snippets.num_rows, 128))

    print('Start to load wiki index from %s' % wiki40b_file_name, time() - st_time)
    quantiser = faiss.IndexFlatL2(128)
    wiki40b_index_flat = faiss.IndexIVFFlat(quantiser, 128, 128, faiss.METRIC_L2)

    print('Train wiki index from %s' % wiki40b_file_name, time() - st_time)
    wiki40b_index_flat.train(wiki40b_passage_reps)
    wiki40b_index_flat.add(wiki40b_passage_reps)

    print('Save wiki index', time() - st_time)
    with open('support_docs/wiki40b_index.bin', 'wb') as f:
        pickle.dump(wiki40b_index_flat, f)

    def query_all(q_set, q_embed_file, filename):
        q_id = [q['q_id'] for q in q_set]
        q_vectors = np.memmap(q_embed_file, dtype='float32', mode='r', shape=(q_set.num_rows, 128))
        support_doc, dense_res_list = query_qa_dense_index(
            q_vectors,
            wiki40b_snippets,
            wiki40b_index_flat,
            st_time=st_time,
            n_results=10
        )
        docs = {'q_id': q_id, 'documents': support_doc, 'doc_res_list': dense_res_list}
        f = open(filename, 'wb')
        pickle.dump(docs, f)
        f.close()
        print('Save to %s' % filename, time() - st_time)

    eli5c = datasets.load_dataset('jsgao/eli5_category')
    query_all(eli5c['train'], 'embeds/eli5c_train.dat', 'support_docs/eli5c_train_docs.dat')
    query_all(eli5c['validation1'], 'embeds/eli5c_val1.dat', 'support_docs/eli5c_val1_docs.dat')
    query_all(eli5c['validation2'], 'embeds/eli5c_val2.dat', 'support_docs/eli5c_val2_docs.dat')
