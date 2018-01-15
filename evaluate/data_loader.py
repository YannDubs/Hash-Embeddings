import os,csv
import torch
from torch.utils.data import Dataset

class AgNews(Dataset):
    def __init__(self,
                 path,
                 maxLength=None,
                 transform=None,
                 train=True):
        r"""`AG's News <http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html>` dataset.
        
        Args:
            path (string): Root directory of dataset.
            maxLength (int,optional): max length of a text.
            transform (callable,optional) A function/transform that takes in a text and returns a
                preprocessed version.
            train (bool, optional): If True, creates dataset from ``train.csv`` else ``test.csv``.
        """
        self.maxWord = 1000000
        self.train = train
        self.path = os.path.join(path, "train.csv" if self.train else "test.csv")
        self.maxLen = float("inf") if maxLength is None else maxLength
        self.label = None
        self.data = None
        self.transform = transform
        self.load()
            
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        X = self.data[idx,:]
        y = self.label[idx]
        return X, y

    def load(self):
        with open(self.path, 'r', encoding="utf-8") as f:
            nRows = sum(1 for row in f)
            label = [None]*nRows
            data = [None]*nRows
            f.seek(0)
            reader = csv.reader(f, delimiter=',', quotechar='"')
            maxLen = 0
            for i, row in enumerate(reader):
                label[i] = int(row[0])
                txt = ' '.join(row[1:])
                if self.transform is not None:
                    txt = self.transform(txt)               
                data[i] = torch.Tensor([word_encoder(w,self.maxWord) for i,w in enumerate(txt.split()) if i < self.maxLen])
                maxLen = max(maxLen,data[i].shape[0])
            self.maxLen = min(self.maxLen,maxLen)
         
        self.data = torch.zeros(nRows, self.maxLen)
        for i,tokenIds in enumerate(data):
            length = tokenIds.shape[0]
            self.data[i,:length] = tokenIds
        self.label = torch.LongTensor(label).view(-1,1)