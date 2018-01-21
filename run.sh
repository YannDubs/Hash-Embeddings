#!/usr/bin/env bash

cd -P -- "$(dirname -- "$0")"

echo '--- Predict with hashembedding for AgNews ---'
PYTHONHASHSEED=0 python -u evaluate/main.py -d ag | tee -a results.txt

echo
echo '--- Predict without hashembedding for AgNews ---'
PYTHONHASHSEED=0 python -u evaluate/main.py -d ag --no-hashembed | tee -a results.txt

echo
echo '--- Predict with hashembedding for AgNews ---'
PYTHONHASHSEED=0 python -u evaluate/main.py -d amazon | tee -a results.txt

echo
echo '--- Predict without hashembedding for AgNews ---'
PYTHONHASHSEED=0 python -u evaluate/main.py -d amazon --no-hashembed | tee -a results.txt

echo
echo '--- Predict with hashembedding for dbpedia ---'
PYTHONHASHSEED=0 python -u evaluate/main.py -d dbpedia | tee -a results.txt

echo
echo '--- Predict without hashembedding for dbpedia ---'
PYTHONHASHSEED=0 python -u evaluate/main.py -d dbpedia --no-hashembed | tee -a results.txt

echo
echo '--- Predict with hashembedding for sogou ---'
PYTHONHASHSEED=0 python -u evaluate/main.py -d sogou | tee -a results.txt

echo
echo '--- Predict without hashembedding for sogou ---'
PYTHONHASHSEED=0 python -u evaluate/main.py -d sogou --no-hashembed | tee -a results.txt

echo
echo '--- Predict with hashembedding for yahoo ---'
PYTHONHASHSEED=0 python -u evaluate/main.py -d yahoo | tee -a results.txt

echo
echo '--- Predict without hashembedding for yahoo ---'
PYTHONHASHSEED=0 python -u evaluate/main.py -d yahoo --no-hashembed | tee -a results.txt

echo
echo '--- Predict with hashembedding for yelp ---'
PYTHONHASHSEED=0 python -u evaluate/main.py -d yelp | tee -a results.txt

echo
echo '--- Predict without hashembedding for yelp ---'
PYTHONHASHSEED=0 python -u evaluate/main.py -d yelp --no-hashembed | tee -a results.txt

echo
echo '--- Predict with hashembedding for yelp-polarity ---'
PYTHONHASHSEED=0 python -u evaluate/main.py -d yelp-polarity | tee -a results.txt

echo
echo '--- Predict without hashembedding for yelp-polarity ---'
PYTHONHASHSEED=0 python -u evaluate/main.py -d yelp-polarity --no-hashembed | tee -a results.txt
