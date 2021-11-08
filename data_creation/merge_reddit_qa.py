import argparse
import gzip
import json
import os
import re
from os.path import join as pjoin
from time import time

from spacy.lang.en import English

# URL match regex
URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""

html_pairs = [
    ("&amp;", " & "),
    ("&quot", ' " '),
    ("&apos", " ' "),
    ("&gt;", " > "),
    ("&lt;", " < "),
]

tokenizer = English().tokenizer


def word_url_tokenize(st, max_len=20480, max_cont_len=512):
    """
    tokenize and extract url from text
    :param st: input text
    :param max_len: max number of words in the input text
    :param max_cont_len: max number of words in the input text
    :return: output text, urls
    """
    stp = ' '.join([w[:max_cont_len] if w[:max_cont_len].count('.') <= 12 else '.' for w in st.split()[:max_len]])
    url_list = list(set(re.findall(URL_REGEX, stp)))
    for i, url in enumerate(url_list):
        stp = stp.replace(url, " URL_%d " % (i,))
    for a, b in html_pairs:
        stp = stp.replace(a, b)
    pre_txt = ' '.join([str(x) for x in tokenizer(stp)])
    return ' '.join(pre_txt.split()), url_list


def load_and_preprocess_qa(file_dir, mode, st_time):
    """
    read and preprocess questions and answers
    :param file_dir: folder that store all downloaded questions or answers
    :param mode: 'Q' for questions, 'A' for answers
    :param st_time: start time of this script
    :return: list of questions or answers
    """
    filenames = [f_name for f_name in os.listdir(file_dir) if f_name.startswith(f'reddit-{mode.lower()}-20')]
    if mode == 'Q':
        fields = ['id', 'link_flair_text', 'score', 'url', 'title', 'selftext']
    else:
        fields = ['id', 'link_id', 'score', 'body']
    reddit_list = []
    for f_name in filenames:
        f = open(pjoin(file_dir, f_name), 'r')
        for i, l in enumerate(f):
            json_dct = json.loads(l)
            reddit_dct = {}
            for k in fields:
                if k in ['title', 'selftext', 'body']:
                    if json_dct[k].lower() in ['[removed]', '[deleted]']:
                        json_dct[k] = ''
                    txt, url_list = word_url_tokenize(json_dct[k])
                    reddit_dct[k] = (' '.join(txt.split()), url_list)
                else:
                    reddit_dct[k] = json_dct[k]
            reddit_list.append(reddit_dct)
        print('Processed %d lines' % len(reddit_list), time() - st_time)
        f.close()
    return reddit_list


def main():
    parser = argparse.ArgumentParser(description='Subreddit QA merger')
    parser.add_argument('-qd', '--question_dir', default='preprocess_data/q', type=str,
                        help='downloaded questions folder name')
    parser.add_argument('-ad', '--answer_dir', default='preprocess_data/a', type=str,
                        help='downloaded answers folder name')
    parser.add_argument('-z', '--compress', action='store_true',
                        help='compress final dataset by gzip')
    args = parser.parse_args()
    st_time = time()
    question_list = load_and_preprocess_qa(args.question_dir, 'Q', st_time)
    print('Select %d questions after preprocessing' % len(question_list), time() - st_time)
    answer_list = load_and_preprocess_qa(args.answer_dir, 'A', st_time)
    print('Select %d answers after preprocessing' % len(answer_list), time() - st_time)
    reddit_list = {}
    category_whitelist = {'Biology', 'Other', 'Technology', 'Physics', 'Chemistry', 'Economics',
                          'Culture', 'Engineering', 'Mathematics', 'Earth Science', 'Psychology'}
    for q_dict in question_list:
        if q_dict['link_flair_text'] not in category_whitelist:
            continue
        reddit_list[q_dict['id']] = {'q_id': q_dict['id'],
                                     'category': q_dict['link_flair_text'],
                                     'title': q_dict['title'][0],
                                     'title_urls': {'url': q_dict['title'][1]},
                                     'selftext': q_dict['selftext'][0],
                                     'selftext_urls': {'url': q_dict['selftext'][1]},
                                     'subreddit': 'explainlikeimfive',
                                     'answers': {}}
    print('Select %d questions by category whitelist' % len(reddit_list), time() - st_time)
    merged_answer = 0
    for a_dict in answer_list:
        link_id = a_dict['link_id'].split('_')[-1]
        if link_id not in reddit_list:
            continue
        if len(a_dict['body'][0].split()) < 8:
            continue
        reddit_list[link_id]['answers'][a_dict['id']] = {'score': a_dict['score'], 'id': a_dict['id'], 'body': a_dict['body']}
        merged_answer += 1
    print('Merged %d answers to questions by id' % merged_answer, time() - st_time)
    start_re = re.compile('\\[?\\s?eli[5f]\\s?]?[:,]?', re.IGNORECASE)
    final_reddit_list = []
    for link_id in reddit_list:
        if len(reddit_list[link_id]['answers'].values()) == 0:
            continue
        reddit_list[link_id]['title'] = start_re.sub('', reddit_list[link_id]['title']).strip()
        sorted_answers = sorted(reddit_list[link_id]['answers'].values(),
                                key=lambda c: (-c['score'], len(c['body'][0].split()), c['id']))
        reddit_list[link_id]['answers'] = {'a_id': [c['id'] for c in sorted_answers],
                                           'text': [c['body'][0] for c in sorted_answers],
                                           'text_urls': [c['body'][1] for c in sorted_answers],
                                           'score': [c['score'] for c in sorted_answers]}
        final_reddit_list.append(reddit_list[link_id])
    print('%d qa pairs remain in final dataset after postprocessing.' % len(final_reddit_list), time() - st_time)
    with open('preprocess_data/eli5.json', 'w') as f:
        json.dump(final_reddit_list, f)
    print('Saved final dataset to preprocess_data/eli5.json', time() - st_time)
    if args.compress:
        with gzip.open('preprocess_data/eli5.json.gz', 'wt', encoding='UTF-8') as f_compress:
            f_compress.write(json.dumps(final_reddit_list))
        print('Saved compressed final dataset to preprocess_data/eli5.json.gz', time() - st_time)


if __name__ == '__main__':
    main()
