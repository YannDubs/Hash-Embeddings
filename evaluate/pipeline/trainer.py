from timeit import default_timer

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from evaluate.pipeline.helpers import train_valid_load


class EarlyStopping:
    """Callable utility returns `True` when should stop the training early due to no improvement.

    Args:
        patience (torch.Tensor): Number of epochs with no improvement after which training will be stopped.
        mode ({"min","max"},optional): Will stop when reaches a (local) `min`/`max`.
        verbose (int,optional): verbosity level in [0,3].

    Example:
        >>> earlyStopping = EarlyStopping(patience=10)
        >>> for epoch in range(50):
        >>>     ... # train
        >>>     accuracy = ...
        >>>     if earlyStopping(accuracy):
        >>>         earlyStopping.on_train_end()
        >>>         break
    """

    def __init__(self, patience=5, mode="max", verbose=1):
        assert mode in {'min', 'max'}, "unkown mode"
        self.wait = 0
        self.best = 0
        self.patience = patience
        self.operator = np.greater if "max" else np.less
        self.epoch = -1
        self.verbose = verbose

    def __call__(self, metric):
        '''Given the current metric returns `True` if should stop early.'''
        if self.operator(metric, self.best):
            self.best = metric
            self.wait = 0
        else:
            self.wait += 1
        self.epoch += 1
        return self.wait >= self.patience

    def on_train_end(self):
        """Function to call just before stopping: i.e prints information."""
        if self.verbose > 0:
            print("Stoping at epoch:{} with patience:{}. Best:{}.".format(self.epoch, self.patience, self.best))


def evaluate_accuracy(dataIter, model):
    """Given a iterator over a `dataset` and a `model`, returns the accuracy."""
    total, correct = 0, 0
    for x, y in dataIter:
        x = Variable(x)
        y = y.squeeze()
        outputs = model(x)
        yHat = outputs.max(1)[1].data
        correct += yHat.cpu().eq(y).sum()
        total += y.size(0)
    return correct / total


class Trainer:
    """Pipeline for training.

    Args:
        model (torch.nn.Module): Model to train.
        criterion (torch.nn.modules.loss._Loss,optional): Loss to optimize for.
        optimizer (toch.optim.optimizer.Optimizer,optional): Optimizer for training the model.
        verbose (int,optional): verbosity level in [0,3].
        seed (int, optional): sets the seed for generating random numbers.
        metric (callable,optional): Metric for evaluation (not optimized).
        isCuda (bool,optional): Whether to use GPU.

    Example:
        >>> trainer = Trainer(model)
        >>> callbacks = [EarlyStopping(patience=10)]
        >>> trainer(train,callbacks=callbacks,validSize=0.1)
        >>> trainer.evaluate(test)
    """

    def __init__(self,
                 model,
                 criterion=nn.CrossEntropyLoss,
                 optimizer=torch.optim.Adam,
                 verbose=3,
                 seed=123,
                 metric="accuracy",
                 isCuda=torch.cuda.is_available()):
        self.model = model
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        self.isCuda = isCuda
        if self.isCuda:
            assert torch.cuda.is_available()
            print("Using CUDA")
            self.model = self.model.cuda()
            torch.cuda.manual_seed(seed)

        self.criterion = criterion
        self.optimizer = optimizer
        self.verbose = verbose
        self.seed = seed
        self.eval_metric = metric
        if metric == "accuracy":
            self.eval_metric = evaluate_accuracy
        self.criterion = criterion()
        self.optimizer = optimizer(model.parameters())

    def _train_valid_split(self, trainDataset, validDataset, validSize, **kwargs):
        if validDataset:
            return (train_valid_load(trainDataset, validSize=0, **kwargs)[0],
                    train_valid_load(validDataset, validSize=0, **kwargs)[0])

        return train_valid_load(trainDataset, validSize=validSize, seed=self.seed, **kwargs)

    def __call__(self,
                 trainDataset,
                 validDataset=None,
                 validSize=0.05,
                 callbacks=[None],
                 batch_size=32,
                 epochs=10,
                 **kwargs):
        """Trains the model of the current pipeline.

        Args:
            trainDataset (torch.utils.data.Dataset): Dataset on which to train the model.
            validDataset (torch.utils.data.Dataset, optional): Dataset used for validation.
            validSize (float, optional): If `validDataset` is `None`, then will randomly select `validSize` of
                `trainDataset` for validation.
            callbacks (list of callable, optional): List of callable functions used to be applied at given stages
                of the training procedure. Ex: `EarlyStopping`.
            batch_size (int,optional): Number of examples per batch.
            epochs (int,optional): Maximum number of epochs.
            kwargs: Additional arguments to the `DataLoaders`.

        Example:
            >>> trainer = Trainer(model)
            >>> callbacks = [EarlyStopping(patience=10)]
            >>> trainer(train,callbacks=callbacks,validSize=0.1)
            >>> trainer.evaluate(test)
        """
        start = default_timer()
        train, valid = self._train_valid_split(trainDataset,
                                               validDataset,
                                               validSize,
                                               batch_size=batch_size,
                                               **kwargs)
        if self.verbose > 0:
            print('Num parameters in model: {}'.format(sum([np.prod(p.size()) for p in self.model.parameters()])))
            print("Train on {} samples, validate on {} samples".format(len(train), len(valid)))

        for epoch in range(epochs):
            for x, y in train:
                if self.isCuda:
                    y = y.cuda()
                x = Variable(x)
                y = Variable(y).squeeze()

                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

            metric = self.eval_metric(valid, self.model)
            if self.verbose > 0 and epoch % (4 - self.verbose) == 0:
                print("Time since start: {:.1f}. Epoch: {}. Loss: {}. Acc: {}.".format((default_timer() - start) / 60, epoch, loss.data[0], metric))

            for callback in callbacks:
                if isinstance(callback, EarlyStopping) and callback(metric):
                    callback.on_train_end()
                    return

    def evaluate(self, test):
        """Evaluates the model on a itraitable `test` dataset."""
        print("Test accuracy:", self.eval_metric(test, self.model))
