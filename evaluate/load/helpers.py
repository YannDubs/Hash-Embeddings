from __future__ import unicode_literals,division,absolute_import,print_function
from builtins import open,str,range

import sys
import string
import hashlib
from collections import Counter

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


# Stoped using because already done in sklearn. Note that this is quicker than in sklearn (~2x) but only works for uni grams yet.
def text_to_word_sequence(txt,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, 
                          split=" ",
                          maxLength=None):
    """Converts a text to a sequence of words (or tokens).

    Args:
        txt (str): Input text (string).
        filters (str,optional): Sequence of characters to filter out.
        lower (bool,optional): Whether to convert the input to lowercase.
        split (bool,optional): Sentence split marker (string).
        maxLength (int,optional): max length of a text. Drops the rest.

    Returns:
        A list of words (or tokens).
    """
    maxLen = float("inf") if maxLength is None else maxLength

    if lower:
        txt = txt.lower()

    if sys.version_info < (3,) and isinstance(text, unicode):
        translate_map = dict((ord(c), unicode(split)) for c in filters)
    else:
        translate_map = str.maketrans(filters, split * len(filters))

    txt = txt.translate(translate_map)
    
    for i,el in enumerate(txt.split(split)):
        if i >= maxLen:
            break
        if el:
            yield el


# Stoped using because already done in sklearn. Note that this is quicker than in sklearn (~2x) but only works for uni grams yet.
def hashing_trick(txt, 
                  n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' ',
                  maxLength=None,
                  mask_zero=True):
    """Converts a text to a sequence of indexes in a fixed-size hashing space.

    Args:
        text (string): Input text.
        n (int): Dimension of the hashing space.
        hash_function (string or callable): if `None` uses python `hash` function, 
            can be 'md5', 'sha1' or any function that takes in input a string and
            returns a int. Note that `hash` is not a stable contray to the others. 
        filters (string,optional): Sequence of characters to filter out.
        lower (bool, optional: Whether to convert the input to lowercase.
        split (string, optional): Sentence split marker (string).
        maxLength (int,optional): max length of a text. Drops the rest.
        mask_zero (bool, optional): whether the 0 input shouldn't be assigned to any word.

    Returns:
        A list of integer word indices (unicity non-guaranteed).
    """
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
    elif hash_function == 'sha1':
        hash_function = lambda w: int(hashlib.sha1(w.encode('utf-8')).hexdigest(), 16)

    seq = text_to_word_sequence(txt,
                                filters=filters,
                                lower=lower,
                                split=split,
                                maxLength=maxLength)

    if mask_zero:
        for w in seq:
            yield hash_function(w) % (n - 1) + 1
    else:
        for w in seq:
            yield hash_function(w) % n


# Stoped using because already done in sklearn. Note that this is much than in sklearn (~2x) but only works for uni grams yet.
class Vocabulary:
    r"""Utility to encode text into token ids. Replaces the `hashing trick` by using a pre defined dictionnary.

    Args:
        num_words (int,optional): Maximum number of tokens to encode (will keep the most common n).
        filters (str,optional): Sequence of characters to filter out.
        lower (bool, optional: Whether to convert the input to lowercase.
        split (string, optional): Sentence split marker (string).
        maxLength (int,optional): max length of a text. Drops the rest.
        mask_zero (bool, optional): whether the 0 input shouldn't be assigned to any word.
    """
    def __init__(self, 
                 num_words=None,
                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True,
                 split=' ',
                 maxLength=None):
        self.word_counts = Counter()
        self.word_index = {}
        self.filters = filters
        self.split = split
        self.lower = lower
        self.maxLen = maxLength
        self.num_words = num_words
        self.is_finish_fitting = False
        self.text_to_word_sequence = lambda txt: text_to_word_sequence(txt,filters=self.filters,lower=self.lower,split=self.split,maxLength=self.maxLen)

    def __getitem__(self, word):
        """Gets the ID associated with the given token."""
        return self.word_index[word]

    def fit_tokenize(self,txt):
        """FIts the dictionnary and replaces the tokens by ID all at once. Cannot keep only the most n common words."""
        assert self.num_words is None, "You can only fit_tokenize if you sum_words=None but num_words={}. Please use online_fit and tokenize separately.".format(self.num_words)
        
        seq = self.text_to_word_sequence(txt)
        for w in seq:
            if w not in self.word_index:
                self.word_index[w] = len(self.word_index) + 1 # never assign zero
            yield self.word_index[w]

    def online_fit(self,txt):
        """Fits the dictionnary in the first pass. Can call multiple times untile you call `finish_fitting`."""
        assert not self.is_finish_fitting, "You canot refit after finishing fitting"
        seq = self.text_to_word_sequence(txt)
        self.word_counts.update(seq)

    def finish_fitting(self):
        """Keeps only the `n` most common words for memory reasons. Once called, you can no longer call `online_fit`."""
        assert not self.is_finish_fitting, "You can only call finish_fitting once"
        
        self.word_index = {w[0]:i+1 for i,w in enumerate(self.word_counts.most_common(self.num_words))}

        self.word_counts = None # free some space
        self.is_finish_fitting = True

    def tokenize(self,txt):
        """Replaces the tokens by their respective IDs in the previously fitted dictionnary."""
        if not self.is_finish_fitting:
            self.finish_fitting()

        seq = self.text_to_word_sequence(txt)
        for w in seq:
            if w in self.word_index:
                yield self.word_index[w]

def train_valid_load(dataset,validSize=0.1,isShuffle=True,seed=123,**kwargs):
    r"""Utility to split a training set into a validation and a training one.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to split.
        validSize (float,optional): Percentage to keep for the validation set. In [0,1].
        isShuffle (bool,optional): Whether should shuffle before splitting.
        seed (int, optional): sets the seed for generating random numbers.
        kwargs: Additional arguments to the `DataLoaders`.

    Returns:
        The train and the valid DataLoader, respectively.
    """
    assert 0 <= validSize <= 1, "validSize:{}. Should be in [0,1]".format(validSize)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
    if validSize == 0:
        return DataLoader(dataset,**kwargs), iter(())
    
    nTrain = len(dataset)
    idcs = np.arange(nTrain)
    splitIdx = int(validSize * nTrain)

    if isShuffle:
        np.random.shuffle(idcs)

    trainIdcs, validIdcs = idcs[splitIdx:], idcs[:splitIdx]

    trainSampler = SubsetRandomSampler(trainIdcs)
    validSampler = SubsetRandomSampler(validIdcs)

    trainLoader = DataLoader(dataset,sampler=trainSampler,**kwargs)

    validLoader = DataLoader(dataset,sampler=validSampler,**kwargs)

    return trainLoader, validLoader
