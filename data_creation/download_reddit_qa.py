import argparse
import bz2
import json
import lzma
import io
import os
import re
import subprocess
from os.path import join as pjoin
from time import time, sleep
import zstandard as zstd

import requests
from bs4 import BeautifulSoup

REDDIT_URL = "https://files.pushshift.io/reddit/"


def get_reddit_backup_urls(mode):
    """
    Parse reddit backups on pushshift.io
    :param mode: 'Q' for questions, 'A' for answers
    :return: dict of (year, month): backup_url
    """
    mode = {'Q': 'submissions', 'A': 'comments'}[mode]
    page = requests.get(REDDIT_URL + mode)
    soup = BeautifulSoup(page.content, 'lxml')
    files = [it for it in soup.find_all(attrs={'class': 'file'})]
    f_urls = [tg.find_all(lambda x: x.has_attr('href'))[0]['href']
              for tg in files if len(tg.find_all(lambda x: x.has_attr('href'))) > 0]
    dict_date_url = {}
    for url_st in f_urls:
        ls = re.findall(r"20[0-9]{2}-[0-9]{2}", url_st)
        if len(ls) > 0:
            yr, mt = ls[0].split('-')
            dict_date_url[(int(yr), int(mt))] = REDDIT_URL + mode + url_st[1:]
    return dict_date_url


def check_download_end(retries):
    for i in range(retries):
        suffix = [f_name.split('.')[-1] for f_name in os.listdir('reddit_tmp')]
        if 'aria2' not in suffix:
            return True
        sleep(10)
    return False


def download(file_urls, ym_list, mode, max_connection=8, max_concurrent=4, retries=4):
    """
    Download compressed files from pushshift.io and preprocess them
    :param file_urls: list of urls on pushshift.io
    :param ym_list: list of (year, month) tuples, same length as file_urls, only use for saved filename
    :param mode: 'Q' for questions, 'A' for answers
    :param max_connection: -x parameter for aria2c
    :param max_concurrent: batch size of download+preprocess and -j parameter for aria2c
    :param retries: maximum retry times when downloads failed
    :return:
    """
    st_time = time()
    for batch in range(0, len(file_urls), max_concurrent):
        f_urls = file_urls[batch: batch + max_concurrent]
        yms = ym_list[batch: batch + max_concurrent]
        download_list_name = 'reddit_tmp/qa_download_files.txt'
        with open(download_list_name, 'w') as f:
            f.writelines('\n'.join(f_urls))
        print('downloading %s-%s %2f' % (yms[0], yms[-1], time() - st_time))
        r = retries
        while r > 0:
            subprocess.run(['aria2c', '-c',
                            '-j', str(max_concurrent),
                            '-x', str(max_connection),
                            '-m', str(retries),
                            '-d', 'reddit_tmp',
                            '-i', download_list_name], stdout=subprocess.PIPE)
            r = 0 if check_download_end(retries) else r - 1
        print('downloaded %s-%s %2f' % (yms[0], yms[-1], time() - st_time))
        local_names = [pjoin('reddit_tmp', url.split('/')[-1]) for url in f_urls]
        for f_name, ym in zip(local_names, yms):
            if f_name.split('.')[-1] == 'xz':
                f = lzma.open(f_name, 'rt')
            elif f_name.split('.')[-1] == 'bz2':
                f = bz2.open(f_name, 'rt')
            elif f_name.split('.')[-1] == 'zst':
                fh = open(f_name, 'rb')
                dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
                stream_reader = dctx.stream_reader(fh)
                f = io.TextIOWrapper(stream_reader, encoding='utf-8')
            lines = []
            print('Preprocessing %s %2f' % (f_name, time() - st_time))
            for i, l in enumerate(f):
                if i % 1000000 == 0:
                    print('read %d lines, found %d' % (i, len(lines)), time() - st_time)
                if '"explainlikeimfive"' in l:
                    lines.append(l)
            if f_name.split('.')[-1] == 'zst':
                fh.close()
            else:
                f.close()
            os.remove(f_name)
            preprocess_lines = []
            if mode == 'Q':
                for l in lines:
                    j = json.loads(l)
                    if j.get('subreddit', '') == 'explainlikeimfive' and \
                            j.get('num_comments', 0) > 0 and j.get('score', 0) > 2:
                        preprocess_lines.append(l)
            elif mode == 'A':
                for l in lines:
                    j = json.loads(l)
                    if j.get('subreddit', '') == 'explainlikeimfive' and \
                            j.get('score', 0) > 2 and len(j.get('body', '').split()) > 2 and \
                            not j.get('body', '').startswith('Your submission has been removed') and\
                            j.get('author', '') != 'AutoModerator' and j['parent_id'] == j['link_id']:
                        preprocess_lines.append(l)
            print('Select %d lines after preprocessing' % len(preprocess_lines), time() - st_time)
            y, m = ym
            with open(f'preprocess_data/reddit-{mode.lower()}-{y}-{m}.json', 'w') as out:
                out.writelines(''.join(preprocess_lines))
            print('Saved %s %2f' % (f_name, time() - st_time))


def year_month_period(start_year, start_month, end_year, end_month):
    """
    Generate all year,month tuple between given time period
    :param start_year: int, start from this year
    :param start_month: int, start from this month
    :param end_year: int, end to this year(inclusive)
    :param end_month: int, end to this month(inclusive)
    :return: list[tuple(year, month)]
    """
    ym_list = []
    for year in range(start_year, end_year + 1):
        st_month = start_month if year == start_year else 1
        ed_month = end_month if year == end_year else 12
        months = range(st_month, ed_month + 1)
        for month in months:
            ym_list.append((year, month))
    return ym_list


def main():
    parser = argparse.ArgumentParser(description='Subreddit QA downloader')
    parser.add_argument('-sy', '--start_year', default=2017, type=int, metavar='N',
                        help='starting year')
    parser.add_argument('-ey', '--end_year', default=2021, type=int, metavar='N',
                        help='end year')
    parser.add_argument('-sm', '--start_month', default=1, type=int, metavar='N',
                        help='starting month')
    parser.add_argument('-em', '--end_month', default=6, type=int, metavar='N',
                        help='end month')
    parser.add_argument('-x', '--parallel', default=4, type=int, metavar='N',
                        help='end month')
    parser.add_argument('-Q', '--questions_only', action='store_true',
                        help='only download submissions')
    parser.add_argument('-A', '--answers_only', action='store_true',
                        help='only download comments')
    args = parser.parse_args()
    dict_date_url_q = get_reddit_backup_urls('Q')
    dict_date_url_a = get_reddit_backup_urls('A')
    year_month_list = year_month_period(args.start_year, args.start_month, args.end_year, args.end_month)
    subprocess.run(['mkdir', 'reddit_tmp'], stdout=subprocess.PIPE)
    subprocess.run(['mkdir', 'preprocess_data'], stdout=subprocess.PIPE)
    q_urls = []
    a_urls = []
    for ym in year_month_list:
        q_urls.append(dict_date_url_q[ym])
        a_urls.append(dict_date_url_a[ym])
    if not args.answers_only:
        download(q_urls, year_month_list, mode='Q', max_concurrent=args.parallel)
    if not args.questions_only:
        download(a_urls, year_month_list, mode='A', max_concurrent=args.parallel)


if __name__ == '__main__':
    main()
