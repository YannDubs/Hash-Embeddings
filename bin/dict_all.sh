#!/usr/bin/env bash

cd -P -- "$(dirname -- "$0")"/../

declare -a datasets=('ag' 'amazon' 'amazon-polarity' 'dbpedia' 'yahoo' 'yelp' 'yelp-polarity')
declare -a nBuckets=(5 100) #(500 10000 50000 100000 150000)
declare -a nEmbeddings=(50 1000) #(10000 25000 50000 300000 500000 1000000)

for dataset in "${datasets[@]}"
do
	echo '========================================='

    echo
    echo '--- Predict with hashembedding for '"$dataset"' ---'
    for nBucket in "${nBuckets[@]}"
    do
    	echo
    	echo "* nBucket = ""$nBucket"
    	PYTHONHASHSEED=0 python -u evaluate/main.py -d "$dataset" -x "hash-embed-dict" --num-buckets "$nBucket" | tee -a results_dict.txt
    done
   
    echo
    echo '--- Predict without hashembedding for '"$dataset"' ---'
    for nEmbedding in "${nEmbeddings[@]}"
    do
    	echo
    	echo "* nEmbedding = ""$nEmbedding"
    	PYTHONHASHSEED=0 python -u evaluate/main.py -d "$dataset" -x "std-embed-dict" --num-embeding "$nEmbedding" | tee -a results_dict.txt
    done
done

