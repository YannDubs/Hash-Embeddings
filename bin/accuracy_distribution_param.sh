#!/usr/bin/env bash

cd -P -- "$(dirname -- "$0")"/../

declare -a datasets=('ag')
file=results_distribution_param.txt

for dataset in "${datasets[@]}"
do
    for i in {1..10}
    do
        seed=$RANDOM

        echo '========================================='

        echo
        echo "Iter = ""$i""; seed = ""$seed"    

        echo
        echo '--- Predict '"$dataset"' hashembedding ---'
        PYTHONHASHSEED=0 python -u evaluate/main.py -d $dataset -s "$seed" -B 100000 -N 1000000 --agg-mode sum  | tee -a $file

        echo
        echo '--- Predict '"$dataset"' hashembedding no append weight ---'
        PYTHONHASHSEED=0 python -u evaluate/main.py -d $dataset -s "$seed" -B 100000 -N 1000000 --agg-mode sum --no-append-weight | tee -a $file
       
        echo
        echo '--- Predict '"$dataset"' hashembedding weighted median ---'
        PYTHONHASHSEED=0 python -u evaluate/main.py -d $dataset -s "$seed" -B 100000 -N 1000000 --agg-mode median | tee -a $file

        echo
        echo '--- Predict '"$dataset"' hashembedding weighted concatenation ---'
        PYTHONHASHSEED=0 python -u evaluate/main.py -d $dataset -s "$seed" -B 100000 -N 1000000 --agg-mode concatenate | tee -a $file
    done
done

