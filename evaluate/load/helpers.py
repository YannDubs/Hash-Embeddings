from __future__ import unicode_literals,division,absolute_import,print_function
from builtins import open,str,range

import sys
import string
import hashlib
from collections import Counter
from itertools import chain

def n_grams(tokens, n=1):
    r"""Returns an itirator over the `n`-grams given a `listTokens`.

    Args:
        tokens (list): List of tokens.
        n (int,optional): N in n-grams.

    Returns:
        Iterator over the n-grams.
    """
    shiftToken = lambda i: (el for j,el in enumerate(tokens) if j>=i)
    shiftedTokens = (shiftToken(i) for i in range(n))
    tupleNGrams = zip(*shiftedTokens)
    return (" ".join(i) for i in tupleNGrams)

def range_ngrams(tokens, ngramRange=(1,2)):
    r"""Returns an itirator over all `n`-grams for n in range(`ngramRange`) given a `listTokens`.

    Args:
        tokens (list): List of tokens.
        ngramRange (tuple,optional): Range of n (n_min,n_max) exclusive n_max .

    Returns:a
        Iterator over the n-grams.
    """
    return chain(*(n_grams(tokens, i) for i in range(*ngramRange)))

def text_to_word_sequence(txt,
                          filters=string.punctuation + '\n\t',
                          lower=True, 
                          rmSingleChar=True,
                          split=" ",
                          maxLength=None):
    """Converts a text to a sequence of words (or tokens).

    Args:
        txt (str): Input text (string).
        filters (str,optional): Sequence of characters to filter out.
        lower (bool,optional): Whether to convert the input to lowercase.
        rmSingleChar (bool,optional): Whether to remove words with a single letter.
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
        if rmSingleChar and len(el) == 1:
            continue
        if i >= maxLen:
            break
        if el:
            yield el

def hashing_trick(txt, 
                  n=None,
                  hash_function=None,
                  filters=string.punctuation + '\n\t', 
                  lower=True,
                  rmSingleChar=True,
                  split=' ',
                  maxLength=None,
                  mask_zero=True,
                  ngramRange=(1,2)):
    """Converts a text to a sequence of indexes in a fixed-size hashing space.

    Args:
        text (string): Input text.
        n (int): Dimension of the hashing space.
        hash_function (string or callable): if `None` uses python `hash` function, 
            can be 'md5', 'sha1' or any function that takes in input a string and
            returns a int. Note that `hash` is not a stable contray to the others. 
        filters (string,optional): Sequence of characters to filter out.
        lower (bool, optional: Whether to convert the input to lowercase.
        rmSingleChar (bool,optional): Whether to remove words with a single letter.
        split (string, optional): Sentence split marker (string).
        maxLength (int,optional): max length of a text. Drops the rest.
        mask_zero (bool, optional): whether the 0 input shouldn't be assigned to any word.
        ngramRange (tuple,optional): Range of n : (n_min,n_max) exclusive n_max .

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
                                maxLength=maxLength,
                                rmSingleChar=rmSingleChar)

    nGrams = range_ngrams(list(seq),ngramRange=ngramRange)

    if not n:
        return (hash_function(token) for token in nGrams)
    if mask_zero:
        return (hash_function(token) % (n - 1) + 1 for token in nGrams)
    else:
        return (hash_function(token) % n for token in nGrams)

class Vocabulary:
    r"""Utility to encode text into token ids. Replaces the `hashing trick` by using a pre defined dictionnary.

    Args:
        num_words (int,optional): Maximum number of tokens to encode (will keep the most common n).
        filters (str,optional): Sequence of characters to filter out.
        lower (bool, optional: Whether to convert the input to lowercase.
        rmSingleChar (bool,optional): Whether to remove words with a single letter.
        split (string, optional): Sentence split marker (string).
        maxLength (int,optional): max length of a text. Drops the rest.
        mask_zero (bool, optional): whether the 0 input shouldn't be assigned to any word.
        ngramRange (tuple,optional): Range of n : (n_min,n_max) exclusive n_max .
    """
    def __init__(self, 
                 num_words=None,
                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True,
                 rmSingleChar=True,
                 split=' ',
                 maxLength=None,
                 mask_zero=True,
                 ngramRange=(1,2)):
        self.word_counts = Counter()
        self.word_index = {}
        self.filters = filters
        self.split = split
        self.lower = lower
        self.maxLen = maxLength
        self.num_words = num_words
        self.is_finish_fitting = False
        self.text_to_word_sequence = lambda txt: text_to_word_sequence(txt,
                                                                        filters=filters,
                                                                        lower=lower,
                                                                        split=split,
                                                                        maxLength=maxLength,
                                                                        rmSingleChar=rmSingleChar)
        self.mask_zero = mask_zero
        self.ngramRange = ngramRange

    def __getitem__(self, word):
        """Gets the ID associated with the given token."""
        return self.word_index[word]

    def fit_tokenize(self,txt):
        """FIts the dictionnary and replaces the tokens by ID all at once. Cannot keep only the most n common words."""
        assert self.num_words is None, "You can only fit_tokenize if you sum_words=None but num_words={}. Please use fit and tokenize separately.".format(self.num_words)
        
        seq = self.text_to_word_sequence(txt)
        nGrams = range_ngrams(list(seq),ngramRange=self.ngramRange)

        for token in nGrams:
            if token not in self.word_index:
                self.word_index[token] = len(self.word_index) + int(self.mask_zero) # never assign zero
            yield self.word_index[token]

    def fit(self,txt):
        """Fits the dictionnary in the first pass. Can call multiple times untile you call `finish_fitting`."""
        assert not self.is_finish_fitting, "You canot refit after finishing fitting"
        seq = self.text_to_word_sequence(txt)
        nGrams = range_ngrams(list(seq),ngramRange=self.ngramRange)
        self.word_counts.update(nGrams)

    def finish_fitting(self):
        """Keeps only the `n` most common words for memory reasons. Once called, you can no longer call `fit`."""
        assert not self.is_finish_fitting, "You can only call finish_fitting once"
        
        inc = int(self.mask_zero) # never assign zero
        self.word_index = {w[0]:i+inc for i,w in enumerate(self.word_counts.most_common(self.num_words))}

        self.word_counts = None # free some space
        self.is_finish_fitting = True

    def tokenize(self,txt):
        """Replaces the tokens by their respective IDs in the previously fitted dictionnary."""
        if not self.is_finish_fitting:
            self.finish_fitting()

        seq = self.text_to_word_sequence(txt)
        nGrams = range_ngrams(list(seq),ngramRange=self.ngramRange)

        return (self.word_index[token] for token in nGrams if token in self.word_index)


