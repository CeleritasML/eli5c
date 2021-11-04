#!/bin/sh

cd $(dirname $0)
cd ../
pwd
python download_reddit_qa.py -Q