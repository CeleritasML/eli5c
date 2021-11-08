import argparse
import gzip
import json
from os.path import join as pjoin
from time import time


def main():
    parser = argparse.ArgumentParser(description='Subreddit QA Splitter')
    parser.add_argument('-d', '--dataset', default='preprocess_data/eli5.json', type=str,
                        help='file path of the full dataset')
    parser.add_argument('-s', '--split', default='preprocess_data/split.json', type=str,
                        help='split criterion of the dataset')
    parser.add_argument('-o', '--output', default='dataset', type=str,
                        help='output folder the split dataset')
    parser.add_argument('-z', '--compress', action='store_true',
                        help='compress final dataset by gzip')
    args = parser.parse_args()
    st_time = time()
    filename = str(args.dataset)
    print('Reading ELI5-Category dataset from %s' % filename, time() - st_time)
    if filename.endswith('.json'):
        f = open(filename, 'r')
    elif filename.endswith('.json.gz'):
        f = gzip.open(filename, 'rt', encoding='UTF-8')
    else:
        raise IOError('Only .json and .json.gz are supported file format of ELI5 Category dataset')
    qa_list = json.load(f)
    f.close()
    print('Loaded ELI5-Category dataset, found %d QA pairs' % len(qa_list), time() - st_time)
    print('Parsing split criterion from %s' % args.split, time() - st_time)
    with open(args.split, 'r') as f_split:
        splits = json.load(f_split)
    print('Split criterion are ', splits, time() - st_time)
    splits = {k: set(category) for k, category in splits.items()}
    train_set = []
    validation_set = []
    test_set = []
    for qa in qa_list:
        if qa['category'] in splits['train']:
            train_set.append(qa)
        elif qa['category'] in splits['validation']:
            validation_set.append(qa)
        elif qa['category'] in splits['test']:
            test_set.append(qa)
    print('Finish splitting, %d QA pairs in training set, %d QA pairs in validation set, %d QA pairs in test set' %
          (len(train_set), len(validation_set), len(test_set)), time() - st_time)
    with open(pjoin(args.output, 'eli5-category-train.json'), 'w') as f:
        json.dump(train_set, f)
        print('Saved training dataset to %s' % f.name, time() - st_time)
    with open(pjoin(args.output, 'eli5-category-validation.json'), 'w') as f:
        json.dump(validation_set, f)
        print('Saved validation dataset to %s' % f.name, time() - st_time)
    with open(pjoin(args.output, 'eli5-category-test.json'), 'w') as f:
        json.dump(test_set, f)
        print('Saved test dataset to %s' % f.name, time() - st_time)
    if args.compress:
        with gzip.open('dataset/eli5-category-train.json.gz', 'wt', encoding='UTF-8') as f_compress:
            f_compress.write(json.dumps(train_set))
            print('Saved compressed training dataset to %s' % f_compress.name, time() - st_time)
        with gzip.open('dataset/eli5-category-validation.json.gz', 'wt', encoding='UTF-8') as f_compress:
            f_compress.write(json.dumps(validation_set))
            print('Saved compressed validation dataset to %s' % f_compress.name, time() - st_time)
        with gzip.open('dataset/eli5-category-test.json.gz', 'wt', encoding='UTF-8') as f_compress:
            f_compress.write(json.dumps(test_set))
            print('Saved compressed test dataset to %s' % f_compress.name, time() - st_time)


if __name__ == '__main__':
    main()
