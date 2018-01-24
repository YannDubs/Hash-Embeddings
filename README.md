# Hash-Embeddings
PyTorch implementation of [Hash Embedding for efficient Representation](https://arxiv.org/abs/1709.03933) (NIPS 2017). Submission to the NIPS Implementation Challenge.

Hash Embedding are a **generalization of the hashing trick** in order to get a larger vocabulary with the same amont of parameters, or in other words it can be used to approximate the hashing trick using less parameters. The hashing trick (in the NLP context) is a popular technique where you use a hash table rather than a dictionnary for the word embeddings, which enables online learning (as the table's size is fixed with respect to the vocabulary size) and often helps against overfitting.

## Install

```bash
# clone repo
pip install -r requirements.txt
# install pytorch : http://pytorch.org/
```

## Use Hash Embeddings
If you only want to use the hashembedding:

```python
from hashembed import HashEmbedding
```

## Evaluate Hash Embedding
* Download and untar the [data](http://goo.gl/JyCnZq) in the `data` folder. If the link check [Xiang Zhang's Crepe directory on github](https://github.com/zhangxiangxiao/Crepe)

### Evaluate single model
* use `python evaluate/main <param>` to run a single experiment. If you want perfect replicabiility use, define the python hash seed with `PYTHONHASHSEED=0 python evaluate/main <param>`.

```

usage: main.py [-h] [-d {ag,amazon,dbpedia,sogou,yahoo,yelp,yelp-polarity}]
               [--no-shuffle] [--no-checkpoint] [--val-loss-callback]
               [-e EPOCHS] [-b BATCH_SIZE] [-v VALIDATION_SIZE] [-s SEED]
               [-p PATIENCE] [-V VERBOSE] [-P [PATIENCE [FACTOR ...]]]
               [--no-cuda] [-w NUM_WORKERS] [--dictionnary]
               [-g [MIN_NGRAM [MAX_NGRAM ...]]]
               [-f [MIN_FATURES [MAX_FATURES ...]]] [--no-hashembed]
               [--append-weight] [-D DIM] [-B NUM_BUCKETS] [-N NUM_EMBEDING]
               [-H NUM_HASH]

PyTorch implementation and evaluation of HashEmbeddings, which uses multiple
hashes to efficiently approximate an Embedding layer.

optional arguments:
  -h, --help            show this help message and exit

Dataset options:
  -d {ag,amazon,dbpedia,sogou,yahoo,yelp,yelp-polarity}, --dataset {ag,amazon,dbpedia,sogou,yahoo,yelp,yelp-polarity}
                        path to training data csv. (default: ag)

Learning options:
  --no-shuffle          Disables shuffling batches when training. (default:
                        False)
  --no-checkpoint       Disables model checkpoint. I.e saving best model based
                        on validation loss. (default: False)
  --val-loss-callback   Whether should monitor the callbacks (early stopping ?
                        decrease LR on plateau/ ... on the loss rather than
                        accuracy on validation set. (default: False)
  -e EPOCHS, --epochs EPOCHS
                        Maximum number of epochs to run for. (default: 300)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size for training. (default: 64)
  -v VALIDATION_SIZE, --validation-size VALIDATION_SIZE
                        Percentage of training set to use as validation.
                        (default: 0.05)
  -s SEED, --seed SEED  Random seed. (default: 1234)
  -p PATIENCE, --patience PATIENCE
                        Patience if early stopping. None means no early
                        stopping. (default: 10)
  -V VERBOSE, --verbose VERBOSE
                        Verbosity in [0,3]. (default: 3)
  -P [PATIENCE [FACTOR ...]], --plateau-reduce-lr [PATIENCE [FACTOR ...]]
                        If specified, if loss did not improve since PATIENCE
                        epochs then multiply lr by FACTOR. [None,None] means
                        no reducing of lr on plateau. (default: [4, 0.5])

Device options:
  --no-cuda             Disables CUDA training, even when have one. (default:
                        False)
  -w NUM_WORKERS, --num-workers NUM_WORKERS
                        Number of subprocesses used for data loading.
                        (default: 0)

Featurizing options:
  --dictionnary         Uses a dictionnary. (default: False)
  -g [MIN_NGRAM [MAX_NGRAM ...]], --ngrams-range [MIN_NGRAM [MAX_NGRAM ...]]
                        Range of ngrams to generate. ngrams in
                        [minNgram,maxNgram[. (default: [1, 9])
  -f [MIN_FATURES [MAX_FATURES ...]], --num-features-range [MIN_FATURES [MAX_FATURES ...]]
                        If specified, during training each phrase will have a
                        random number of features in range
                        [minFeatures,maxFeatures[. None if take all. (default:
                        [4, 100])

Embedding options:
  --no-hashembed        Uses the default embedding. (default: False)
  --append-weight       Whether to append the importance parameters. (default:
                        False)
  -D DIM, --dim DIM     Dimension of word vectors. Higher improves downstream
                        task for fixed vocabulary size. (default: 20)
  -B NUM_BUCKETS, --num-buckets NUM_BUCKETS
                        Number of buckets in the shared embedding table.
                        Higher improves approximation quality. (default:
                        1000000)
  -N NUM_EMBEDING, --num-embeding NUM_EMBEDING
                        Number of rows in the importance matrix. Approximate
                        the number of rows in a usual embedding. Higher will
                        increase possible vocabulary size. (default: 10000000)
  -H NUM_HASH, --num-hash NUM_HASH
                        Number of different hashes to use. Higher improves
                        approximation quality. (default: 2)
 ```

### All Experiments
 * Run all the experiments like in the paper with `./run.sh`. *Warning: computationally very intensive.*

## Explanation

In order to understand the advantages of Hash Embeddings and how they work it is probably a good idea to review and compare to a usual dictionnary and to the *hashing trick*.

Notation:
scalar *a*, matrix *M*, *i*th row in a matrix: ***m**=M[i]*, row vector ***v** = (v^1, ..., v^i, ...)*  , function *f()*.

-- General --
* *w_i* : word or token *i*.
* *d* : dimension of each word embedding.
* *E* : table containing all the word embeddings. Size: *n\*d*.
* ***e_w*** : vector embedding of word *w* (word embedding).
* *M[i]* : looks up index *i* in matrix *M*, returns the value in the table.
* *h(x)* : hash function returns the associated index with *x*.

-- Hash Embeddings --
* *u_i(x)* : universal hash function *i* (i.e "sampled hash function") returns the associated index with *x*.
* *k*: number of hash functions.
* *E* : table containing all the word embeddings. Size: *b\*d*.
* *P*: matrix containing the weight for doing the weighted average of the *k* embeddings. Size: *n\*k*. 
* *C_w*: matrix containing the *k* different word embeddings for word *w*. Size: *k\*d*. 

Nota Bene: In the paper *n* for hash embeddings is called *K* but to differentiate with *k* and for consistencz with teh hashing trick I use *n*.


### In Short

* **Usual Dictionnary**: 
    * Pretraining step: Loop through the corpus once and count how many times you see each words. Keep the *n* most common words. Initialize *E*.
    * Training step: loop over word *w* in corpus and update associated embedding in *E*: ***e_w** = E[w]* (size *1\*d*).
    * Online Training: hard.
    * Number of trainable parameters
* **Hashing Trick**:
    * Pretraining step: initalize *E*.
    * Training step: loop over word *w* in corpus and update associated embedding in *E* : ***e_w** = E[H(w)]* (size *1\*d*).
    * Online Training: trivially continue training.
* **Hashing Trick**:
    * Pretraining step: initalize *E* and *P*.
    * Training step: loop over word *w* in corpus and update *P* and *E* such that : ***p_w** = P[u_i(w)]* (size *1\*k*) and ***C_w** = (E[u_1(w)],...,E[u_k(w)])* (size *k\*d*) and ***e_w** = p_w \* C_w* (size *1\*d*).
    * Online Training: trivially continue training.

As we say, a picture is worth a million words. Let's save both of us some time :) :

![Alt text](images/embeddings_explanation.png)

### In long 




## Improvements



## Results
* Datasets : *from Table 1 in the paper.*

| Dataset                | #Train | #Test | #Classes | Task                        |
| :--------------------- |:-----: | :---: |:--------:| :---------------------------|
| AG’s news              | 120k   | 7.6k  |4         | English news categorization |
| DBPedia                | 450k   | 70k   |14        | Ontology classification     |
| Yelp Review Polarity   | 560k   | 38k   |2         | Sentiment analysis          |
| Yelp Review Full       | 560k   | 50k   |5         | Sentiment analysis          |
| Yahoo! Answers         | 650k   | 60k   |10        | Topic classification        |
| Amazon Review Full     | 3000k  | 650k  |5         | Sentiment analysis          |
| Amazon Review Polarity | 3600k  | 400k  |2         | Sentiment analysis          |

* Preprocessing : 
    * Remove punctuation, lowercase, remove single characters. Note that I used the default Keras preprocessing as it's the papers framework, but I now realized it was written that they only remove punctuation, I didn't have the time to rerun everything.
    * Converts to n-grams. 
    * Select randomly between 4-100 n-grams as input (~dropout). 
* Training:
    * Sum n-gram embedding to make phrase emebdding.
    * Adam optimizer.
    * Learning rate = 0.001.
    * Early stooping with patience = 10.
    * 5% of train set used for validation.

In order to compare to the results in the paper I ran the same 3 experiments:
1. **Without a dictionnary**: embedding followed by a softmax.
    * Standard Embeddings : 
        * Hyper parameters: n = 10^7*, *d = 20*, ngram range : *[1,3[*. 
        * Number of trainable parameters : *200 * 10^6*
    * Hash Embeddings : 
        * Hyper parameters: n = 10^7*, *k = 2*, *b = 10^6*, *d = 20*, ngram range : *[1,3[*. 
        * Number of trainable parameters : *40 * 10^6*.
2. **With a dictionnary**: embedding followed by 3 fully connected layers with *1000* hidden units and ReLu activation ends in softmax layer. With bath normalization.
    * Standard Embeddings : 
        * Hyper parameters: *n = cross-validate([10K, 25K, 50K, 300K, 500K, 1M])*, *d = 200*, ngram range : *[1,10[*. 
        * Number of trainable parameters : ...
    * Hash Embeddings : 
        * Hyper parameters: *n = 10^6*, *k = 2*, *b = cross-validate([500, 10K, 50K, 100K, 150K])*, *d = 200*, ngram range : *[1,10[*. 
        * Number of trainable parameters : ...
    * Hash Embedding Ensemble:
        * Averages 10 models with different seed and between 1 and 3 fully connected layers.
        * Hyper parameters: *n = 10^6*, *k = 2*, *b = 50 000*, *d = 200*, ngram range : *[1,10[*. 
        * Number of trainable parameters : ...

### My Results

| Model                  | **No Dict.** &<br/>Hash Emb. | **No Dict.** &<br/>Std Emb. | **Dict.** &<br/>Hash Emb.| **Dict.** &<br/>Std Emb. | **Dict.** &<br/>Ensemble Hash Emb. |
| :--------------------- |:---------:|:-------: |:--------:| :-------:|:------------------:|
| AG’s news              | 92.1      | 91.9     |91.5      | 91.7     | 92.0               |
| DBPedia                | 60.0      | 58.3     |59.4      | 58.5     | 60.5               |
| Yelp Review Polarity   | 98.5      | 98.6     |98.7      | 98.6     | 98.8               |
| Yelp Review Full       | 72.3      | 72.3     |71.3      | 65.8     | 72.9               |
| Yahoo! Answers         | 63.8      | 62.6     |62.6      | 61.4     | 62.9               |
| Amazon Review Full     | 94.4      | 94.2     |94.7      | 93.6     | 94.7               |
| Amazon Review Polarity | 95.9      | 95.5     |95.8      | 95.0     | 95.7               |

### Paper's Results

*from Table 2 in the paper:*

| Model                  | **No Dict.** &<br/>Hash Emb. | **No Dict.** &<br/>Std Emb. | **Dict.** &<br/>Hash Emb.| **Dict.** &<br/>Std Emb. | **Dict.** &<br/>Ensemble Hash Emb. |
| :--------------------- |:---------:|:-------: |:--------:| :-------:|:------------------:|
| AG’s news              | 92.4      | 92.0     |91.5      | 91.7     | 92.0               |
| DBPedia                | 60.0      | 58.3     |59.4      | 58.5     | 60.5               |
| Yelp Review Polarity   | 98.5      | 98.6     |98.7      | 98.6     | 98.8               |
| Yelp Review Full       | 72.3      | 72.3     |71.3      | 65.8     | 72.9               |
| Yahoo! Answers         | 63.8      | 62.6     |62.6      | 61.4     | 62.9               |
| Amazon Review Full     | 94.4      | 94.2     |94.7      | 93.6     | 94.7               |
| Amazon Review Polarity | 95.9      | 95.5     |95.8      | 95.0     | 95.7               |


