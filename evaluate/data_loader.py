from __future__ import unicode_literals,division,absolute_import,print_function
from builtins import open,str,range

import os,csv
import torch
from torch.utils.data import Dataset

from evaluate.helpers import hashing_trick, Vocabulary

class CrepeDataset(Dataset):
    r"""Class for making general datasets of Xiang Zhang's Crepe format `https://github.com/zhangxiangxiao/Crepe`. 
        Dataset `AG's News <http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html>` dataset.
        
        Args:
            path (string): Root directory of dataset.
            maxWord (int,optional): max number of words if hash. If `None` will use a dictionnary.
            maxLength (int,optional): max length of a text.
            train (bool, optional): If True, creates dataset from ``train.csv`` else ``test.csv``.
            **kwargs: Additional arguments to the hashing trick. Default: [ython hash / remove punctuation / lowercase / whitespace.
        """
    def __init__(self,
                 path,
                 maxWord=None,
                 maxLength=None,
                 train=True,
                 **kwargs):
        self.train = train
        self.path = os.path.join(path, "train.csv" if self.train else "test.csv")
        self.maxLen = float("inf") if maxLength is None else maxLength
        self.maxWord = maxWord
        if self.maxWord is None:
            self.vocab = Vocabulary(maxLength=self.maxLen,**kwargs)
            self.hashing_trick = self.vocab.fit_tokenize
        else:
            self.hashing_trick = lambda txt: hashing_trick(txt,self.maxWord,maxLength=self.maxLen,**kwargs)
        self.label = None
        self.data = None
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
                label[i] = int(row[0]) - 1 # because labels start from 1qq in these dataset
                txt = ' '.join(row[1:])   
                data[i] = torch.LongTensor(list(self.hashing_trick(txt)))        
                maxLen = max(maxLen,data[i].shape[0])
            self.maxLen = min(self.maxLen,maxLen)
         
        self.data = torch.zeros(nRows, self.maxLen).type(torch.LongTensor)
        for i,tokenIds in enumerate(data):
            length = tokenIds.shape[0]
            self.data[i,:length] = tokenIds
        self.label = torch.LongTensor(label).view(-1,1)

class AgNews(CrepeDataset):
    def __init__(self,
                 classes={'World': 0, 'Sports': 1, 'Business': 2, 'Sci/Tech': 3},
                 path="../data/ag_news_csv",
                 **kwargs):
        r"""`AG's News <http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html>` dataset.

        The AG's news topic classification dataset is constructed by choosing 4 largest classes from the original corpus. 
        Each class contains 30,000 training samples and 1,900 testing samples. The total number of training samples is 120,000 and testing 7,600.
        classes = {'World': 0, 'Sports': 1, 'Business': 2, 'Sci/Tech': 3}.
        
        Args:
            classes (dict,optional): map class label to label in the data.
            path (string,optional): Root directory of dataset.
            maxLength (int,optional): max length of a text.
            transform (callable,optional) A function/transform that takes in a text and returns a
                preprocessed version.
            train (bool, optional): If True, creates dataset from ``train.csv`` else ``test.csv``.
        """
        super(AgNews,self).__init__(path,**kwargs)
        self.classes = classes

class AmazonReviewPolarity(CrepeDataset):
    r"""`Amazon Review Polarity Dataset` dataset.

        The Amazon reviews polarity dataset is constructed by taking review score 1 and 2 as negative, and 4 and 5 as positive.
         Samples of score 3 is ignored. In the dataset, class 1 is the negative and class 2 is the positive. 
         Each class has 1,800,000 training samples and 200,000 testing samples.
        
        Args:
            path (string,optional): Root directory of dataset.
            maxWord (int,optional): max number of words if hash. If `None` will use a dictionnary.
            maxLength (int,optional): max length of a text.
            transform (callable,optional) A function/transform that takes in a text and returns a
                preprocessed version.
            train (bool, optional): If True, creates dataset from ``train.csv`` else ``test.csv``.
        """
    def __init__(self,
                 path="../data/amazon_review_polarity_csv",
                 **kwargs):
        super(AmazonReviewPolarity,self).__init__(path,**kwargs)

class DbPedia(CrepeDataset):
    r"""`DBPedia Ontology Classification Dataset` dataset.

        The DBpedia ontology classification dataset is constructed by picking 14 non-overlapping classes from DBpedia 2014. 
        They are listed in classes.txt. From each of thse 14 ontology classes, we randomly choose 40,000 training samples 
        and 5,000 testing samples. Therefore, the total size of the training dataset is 560,000 and testing dataset 70,000.
        classes = {'Company': 0, 'EducationalInstitution': 1, 'Artist': 2, 'Athlete': 3, 'OfficeHolder': 4, 'MeanOfTransportation': 5, 
                  'Building': 6, 'NaturalPlace': 7, 'Village': 8, 'Animal': 9, 'Plant': 10, 'Album': 11, 'Film': 12, 'WrittenWork': 13}.
        
        Args:
            classes (dict,optional): map class label to label in the data.
            path (string,optional): Root directory of dataset.
            maxWord (int,optional): max number of words if hash. If `None` will use a dictionnary.
            maxLength (int,optional): max length of a text.
            transform (callable,optional) A function/transform that takes in a text and returns a
                preprocessed version.
            train (bool, optional): If True, creates dataset from ``train.csv`` else ``test.csv``.
        """
    def __init__(self,
                 classes = {'Company': 0, 'EducationalInstitution': 1, 'Artist': 2, 'Athlete': 3, 'OfficeHolder': 4, 'MeanOfTransportation': 5, 
                            'Building': 6, 'NaturalPlace': 7, 'Village': 8, 'Animal': 9, 'Plant': 10, 'Album': 11, 'Film': 12, 'WrittenWork': 13},
                 path="../data/dbpedia_csv",
                 **kwargs):
        super(DbPedia,self).__init__(path,**kwargs)
        self.classes = classes

class SogouNews(CrepeDataset):
    r"""`Sogou News Topic Classification Dataset <http://www.sogou.com/labs/dl/ca.html and http://www.sogou.com/labs/dl/cs.html>` dataset.

        The Sogou news topic classification dataset is constructed by manually labeling each news article according to its URL, 
        which represents roughly the categorization of news in their websites. We chose 5 largest categories for the dataset, 
        each having 90,000 samples for training and 12,000 for testing. The Pinyin texts are converted using pypinyin combined 
        with jieba Chinese segmentation system. In total there are 450,000 training samples and 60,000 testing samples.
        classes = {'sports': 0, 'finance': 1, 'entertainment': 2, 'automobile': 3, 'technology': 4}.

        Args:
            classes (dict,optional): map class label to label in the data.
            path (string,optional): Root directory of dataset.
            maxWord (int,optional): max number of words if hash. If `None` will use a dictionnary.
            maxLength (int,optional): max length of a text.
            transform (callable,optional) A function/transform that takes in a text and returns a
                preprocessed version.
            train (bool, optional): If True, creates dataset from ``train.csv`` else ``test.csv``.
        """
    def __init__(self,
                 classes = {'sports': 0, 'finance': 1, 'entertainment': 2, 'automobile': 3, 'technology': 4},
                 path="../data/sougou_news_csv",
                 **kwargs):
        super(SogouNews,self).__init__(path,**kwargs)
        self.classes = classes

class YahooAnswers(CrepeDataset):
    r"""`Yahoo! Answers Topic Classification Dataset` dataset.

        The Yahoo! Answers topic classification dataset is constructed using 10 largest main categories. Each class contains 
        140,000 training samples and 6,000 testing samples. Therefore, the total number of training samples is 1,400,000 and 
        testing samples 60,000 in this dataset. From all the answers and other meta-information, we only used the best answer 
        content and the main category information.
        classes = {'Society & Culture': 0, 'Science & Mathematics': 1, 'Health': 2, 'Education & Reference': 3, 
                   'Computers & Internet': 4, 'Sports': 5, 'Business & Finance': 6, 'Entertainment & Music': 7, 
                   'Family & Relationships': 8, 'Politics & Government': 9}.

        Args:
            classes (dict,optional): map class label to label in the data.
            path (string,optional): Root directory of dataset.
            maxWord (int,optional): max number of words if hash. If `None` will use a dictionnary.
            maxLength (int,optional): max length of a text.
            transform (callable,optional) A function/transform that takes in a text and returns a
                preprocessed version.
            train (bool, optional): If True, creates dataset from ``train.csv`` else ``test.csv``.
        """
    def __init__(self,
                 classes = {'Society & Culture': 0, 'Science & Mathematics': 1, 'Health': 2, 'Education & Reference': 3, 
                            'Computers & Internet': 4, 'Sports': 5, 'Business & Finance': 6, 'Entertainment & Music': 7, 
                            'Family & Relationships': 8, 'Politics & Government': 9},
                 path="../data/yahoo_answers_csv",
                 **kwargs):
        super(YahooAnswers,self).__init__(path,**kwargs)
        self.classes = classes

class YelpReview(CrepeDataset):
    r"""`Yelp Review Full Star Dataset` dataset.

        The Yelp reviews full star dataset is constructed by randomly taking 130,000 training samples and 10,000 testing samples for each
         review star from 1 to 5. In total there are 650,000 trainig samples and 50,000 testing samples.

        Args:
            path (string,optional): Root directory of dataset.
            maxWord (int,optional): max number of words if hash. If `None` will use a dictionnary.
            maxLength (int,optional): max length of a text.
            transform (callable,optional) A function/transform that takes in a text and returns a
                preprocessed version.
            train (bool, optional): If True, creates dataset from ``train.csv`` else ``test.csv``.
        """
    def __init__(self,
                 path="../data/yelp_review_full_csv",
                 **kwargs):
        super(YelpReview,self).__init__(path,**kwargs)

class YelpReviewPolarity(CrepeDataset):
    r"""`Yelp Review Full Star Dataset` dataset.

        The Yelp reviews polarity dataset is constructed by considering stars 1 and 2 negative, and 3 and 4 positive. For each polarity
        280,000 training samples and 19,000 testing samples are take randomly. In total there are 560,000 trainig samples and 38,000 
        testing samples. Negative polarity is class 1, and positive class 2.

        Args:
            path (string,optional): Root directory of dataset.
            maxWord (int,optional): max number of words if hash. If `None` will use a dictionnary.
            maxLength (int,optional): max length of a text.
            transform (callable,optional) A function/transform that takes in a text and returns a
                preprocessed version.
            train (bool, optional): If True, creates dataset from ``train.csv`` else ``test.csv``.
        """
    def __init__(self,
                 path="../data/yelp_review_polarity_csv",
                 **kwargs):
        super(YelpReview,self).__init__(path,**kwargs)