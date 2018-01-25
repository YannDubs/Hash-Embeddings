#!/usr/bin/env bash

cd -P -- "$(dirname -- "$0")"/../

for i in {1..10}
do
    seed=$RANDOM

    echo '========================================='

    echo
    echo "Iter = ""$i""; seed = ""$seed"    

    echo
    echo '--- Predict Ag with standard hashembedding ---'
    PYTHONHASHSEED=0 python -u evaluate/main.py -d ag -s "$seed" --old-hashembed | tee -a results_distribution.txt

    echo
    echo '--- Predict Ag with improved hashembedding ---'
    PYTHONHASHSEED=0 python -u evaluate/main.py -d ag -s "$seed" | tee -a results_distribution.txt
   
    echo
    echo '--- Predict Ag without hashembedding ---'
    PYTHONHASHSEED=0 python -u evaluate/main.py -d ag -s "$seed" --no-hashembed | tee -a results_distribution.txt
done

