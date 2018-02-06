import os
import random

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from evaluate.load.dataset import get_dataset
from evaluate.load.dataset import get_path


def train_valid_load(dataset, validSize=0.1, isShuffle=True, seed=123, **kwargs):
    r"""Utility to split a training set into a validation and a training one.

    Note:
        This shouldn't be used if the train and test data are prprocessed differently.
        E.g. if you use dropout or a dictionnary for word embeddings.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to split.
        validSize (float,optional): Percentage to keep for the validation set. In [0,1}.
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
        return DataLoader(dataset, **kwargs), iter(())

    nTrain = len(dataset)
    idcs = np.arange(nTrain)
    splitIdx = int(validSize * nTrain)

    if isShuffle:
        np.random.shuffle(idcs)

    trainIdcs, validIdcs = idcs[splitIdx:], idcs[:splitIdx]

    trainSampler = SubsetRandomSampler(trainIdcs)
    validSampler = SubsetRandomSampler(validIdcs)

    trainLoader = DataLoader(dataset, sampler=trainSampler, **kwargs)

    validLoader = DataLoader(dataset, sampler=validSampler, **kwargs)

    return trainLoader, validLoader


def split_file(file, out1, out2, percentage=0.75, isShuffle=True, seed=123):
    """Splits a file in 2 given the approximate `percentage` of the large file."""
    random.seed(seed)
    with open(file, 'r', encoding="utf-8") as fin, \
            open(out1, 'w', encoding="utf-8") as foutBig, \
            open(out2, 'w', encoding="utf-8") as foutSmall:

        nLines = sum(1 for line in fin)
        fin.seek(0)
        nTrain = int(nLines * percentage)
        nValid = nLines - nTrain

        i = 0
        for line in fin:
            r = random.random() if isShuffle else 0  # so that always evaluated to true when not isShuffle
            if (i < nTrain and r < percentage) or (nLines - i > nValid):
                foutBig.write(line)
                i += 1
            else:
                foutSmall.write(line)


def train_valid_test_datasets(datasetId, validSize=None, isHashingTrick=True, specificArgs={'dictionnary': ['num_words']}, **kwargs):
    r"""Loads the train and test datasets given the identifier.

    Args:
        datasetId (string in {ag,amazon,dbpedia,sogou,yahoo,yelp,yelp-polarity}): Dataset identifier.
        validSize (float,optional): Percentage to keep for the validation set. In ]0,1[. If none, doesn't make a validation set.
        isHashingTrick (bool,optional): Whether to use the `hashing trick` rather than a dictionnary.
        specificArgs (dict,optional): Dictionnary specifying which arguments should be used only a certain context.
            Keys are the context identifier and values are a list of string of arguments names.
        kwargs: Additional arguments to the `CrepeDataset` constructor.

    Returns:
        The (train,valid,test) Datasets. Valid is None if validSize==None.
    """
    def rm_specifc_args(identifier, specificArgs, kwargs):
        """Removes arguments specific to a flag in the kwargs."""
        def flatten(l): return (item for sublist in l for item in sublist)
        argsToRm = flatten((v for k, v in specificArgs.items() if k != identifier))
        for arg in argsToRm:
            if arg in kwargs:
                kwargs.pop(arg)
        return kwargs

    Dataset = get_dataset(datasetId)
    valid = None

    if validSize:
        def to_root(x): return os.path.join(get_path(datasetId), x)
        fileTrain = 'train.tmp'
        fileValid = 'valid.tmp'
        split_file(to_root('train.csv'), to_root(fileTrain), to_root(fileValid), percentage=1 - validSize, isShuffle=True)

    if isHashingTrick:
        kwargs = rm_specifc_args('hashingTrick', specificArgs, kwargs)
        if validSize:
            train = Dataset(file=fileTrain, train=True, isHashingTrick=True, **kwargs)
            valid = Dataset(file=fileValid, train=False, isHashingTrick=True, **kwargs)
        else:
            train = Dataset(train=True, isHashingTrick=True, **kwargs)
        test = Dataset(train=False, isHashingTrick=True, **kwargs)

    else:
        kwargs = rm_specifc_args('dictionnary', specificArgs, kwargs)
        if validSize:
            train = Dataset(file=fileTrain, train=True, isHashingTrick=False, **kwargs)
            valid = Dataset(file=fileValid, train=False, isHashingTrick=False, trainVocab=train.vocab, **kwargs)
        else:
            train = Dataset(train=True, isHashingTrick=False, **kwargs)
        test = Dataset(train=False, isHashingTrick=False, trainVocab=train.vocab, **kwargs)

    return train, valid, test
