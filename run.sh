#!/usr/bin/env bash

cd -P -- "$(dirname -- "$0")"

declare -a datasets=('ag' 'amazon' 'amazon-polarity' 'dbpedia' 'yahoo' 'yelp' 'yelp-polarity')

for dataset in "${datasets[@]}"
do
    echo '--- Predict with hashembedding for '"$dataset"' ---'
    PYTHONHASHSEED=0 python -u evaluate/main.py -d "$dataset" | tee -a results.txt
   
    echo
    echo '--- Predict without hashembedding for '"$dataset"' ---'
    PYTHONHASHSEED=0 python -u evaluate/main.py -d "$dataset" --no-hashembed | tee -a results.txt
done

