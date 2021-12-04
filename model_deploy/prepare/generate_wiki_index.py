import pickle
from time import time

import datasets
import numpy as np
import faiss

if __name__ == '__main__':
    st_time = time()
    wiki40b_snippets = datasets.load_dataset('wiki_snippets', name='wiki40b_en_100_0')['train']
    wiki40b_file_name = 'wiki40b.dat'
    wiki40b_passage_reps = np.memmap(wiki40b_file_name, dtype='float32',
                                     mode='r', shape=(wiki40b_snippets.num_rows, 128))

    print('Start to load wiki index from %s' % wiki40b_file_name, time() - st_time)
    quantiser = faiss.IndexFlatL2(128)
    wiki40b_index_flat = faiss.IndexIVFFlat(quantiser, 128, 128, faiss.METRIC_L2)

    print('Train wiki index from %s' % wiki40b_file_name, time() - st_time)
    wiki40b_index_flat.train(wiki40b_passage_reps)
    wiki40b_index_flat.add(wiki40b_passage_reps)

    print('Save wiki index', time() - st_time)
    with open('../models/wiki40b_index.bin', 'wb') as f:
        pickle.dump(wiki40b_index_flat, f)
