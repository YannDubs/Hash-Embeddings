from __future__ import unicode_literals,division,absolute_import,print_function
from builtins import open,str,range

import sys

import string
import hashlib
from collections import Counter

# Modified from Keras, to be able to reproduce the results of the paper.
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

# Modified from Keras, to be able to reproduce the results of the paper.
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
        text: Input text (string).
        n: Dimension of the hashing space.
        hash_function: if `None` uses python `hash` function, can be 'md5', 'sha1' 
            or any function that takes in input a string and returns a int.
            Note that `hash` is not a stable hashing function, so
            it is not consistent across different runs, while 'md5'
            is a stable hashing function.
        filters (str,optional): Sequence of characters to filter out.
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

class Vocabulary(object):
    def __init__(self, 
                 num_words=None,
                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True,
                 split=' ',
                 maxLength=None,
                 oov_token=None,
                 **kwargs):
        self.word_counts = Counter()
        self.word_index = {}
        self.filters = filters
        self.split = split
        self.lower = lower
        self.maxLen = maxLength
        self.num_words = num_words
        self.oov_token = oov_token
        self.is_finish_fitting = False

    def __getitem__(self, word):
        return self.word_index[word]

    def fit_tokenize(self,txt):
        assert self.num_words is None, "You can only fit_tokenize if you sum_words=None but num_words={}. Please use online_fit and tokenize separately.".format(self.num_words)
        
        seq = text_to_word_sequence(txt,filters=self.filters,lower=self.lower,split=self.split,maxLength=self.maxLen)
        for w in seq:
            if w not in self.word_index:
                self.word_index[w] = len(self.word_index) + 1 # never assign zero
            yield self.word_index[w]

    def online_fit(self,txt):
        assert not self.is_finish_fitting, "You canot refit after finishing fitting"
        seq = text_to_word_sequence(txt,filters=self.filters,lower=self.lower,split=self.split,maxLength=self.maxLen)
        self.word_counts.update(seq)

    def finish_fitting(self):
        assert not self.is_finish_fitting, "You can only call finish_fitting once"
        
        self.word_index = {w[0]:i+1 for i,w in enumerate(self.word_counts.most_common(self.num_words))}
        
        if self.oov_token is not None:
            if self.oov_token not in self.word_index:
                self.word_index[self.oov_token] = len(self.word_index) + 1

        self.word_counts = None # free some space
        self.is_finish_fitting = True

    def tokenize(self,txt):
        if not self.is_finish_fitting:
            self.finish_fitting()

        seq = text_to_word_sequence(txt,filters=self.filters,lower=self.lower,split=self.split,maxLength=self.maxLen)
        for w in seq:
            if w in self.word_index:
                yield self.word_index[w]
