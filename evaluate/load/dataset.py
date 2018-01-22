import os,csv
import numpy as np
import torch
from torch.utils.data import Dataset

from evaluate.load.helpers import hashing_trick, Vocabulary

class CrepeDataset(Dataset):
    r"""Class for making general datasets of Xiang Zhang's Crepe format `https://github.com/zhangxiangxiao/Crepe`. 
        
        Args:
            path (string): Root directory of dataset.
            file (string,optional): Filename to append to the path. If none then will use ``train.csv`` or ``test.csv``.
            train (bool,optional): If True, creates dataset from ``train.csv`` else ``test.csv``. 
                If false has to give the train `CrepeDataset`.
            isHashingTrick (bool,optional): Whether to use the `hashing trick` rather than a dictionnary.
            nFeaturesRange (tuple,optional): (minFeatures,maxFeatures) If given, each phrase will have a 
                random number of features K extracted. WHere k in range [minFeatures,maxFeatures[. 
                This can be seen as dropout. Only when training.
            trainVocab (Vocabulary,optional): Vocabulary trained on the train set. Mandatory if not `isHashingTrick` and not `train`.
            seed (int, optional): sets the seed for generating random numbers.
            **kwargs: Additional arguments to the vectorizer : `hashing_trick` or `Vocabulary`. 
                Default `hashing_trick`: n=None, hash_function=None, filters=string.punctuation, lower=True,
                                         rmSingleChar=True, split=' ', maxLength=None, mask_zero=True, ngramRange=(1,2).
                Default `Vocabulary`: n=None, hash_function=None, filters=string.punctuation, lower=True,
                                      rmSingleChar=True, split=' ', maxLength=None, mask_zero=True, ngramRange=(1,2).
        """
    def __init__(self,
                 path,
                 file=None,
                 train=True,
                 isHashingTrick=True,
                 nFeaturesRange=None,
                 trainVocab=None,
                 seed=1234,
                 **kwargs):
        assert (nFeaturesRange is None or (len(nFeaturesRange) == 2 and nFeaturesRange[1] > nFeaturesRange[0] >= 1), 
                "nFeaturesRange has to be a tuple: (minFeatures,maxFeatures) not {}".format(nFeaturesRange))

        np.random.seed(seed)
        self.train = train
        self.path = os.path.join(path, file if file is not None else "train.csv" if self.train else "test.csv")
        self.isHashingTrick = isHashingTrick
        if self.isHashingTrick:
            self.to_ids = lambda txt: hashing_trick(txt,**kwargs) 
        else:
            if self.train:
                self.vocab = Vocabulary(**kwargs)
                self.fit = self.vocab.fit
            else:
                assert trainVocab is not None, "When both train and HashingTrick are false you have to give a vocabulary. Ex : train.vocab"
                self.vocab = trainVocab
                self.fit = lambda x: None
            self.to_ids = self.vocab.tokenize
            #self.vectorizer = HashingVectorizer(n_features=maxFeatures,norm=None,**kwargs) if isHashingTrick else CountVectorizer(max_features=maxFeatures,**kwargs)
        self.nFeaturesRange = nFeaturesRange
        self.maxLength = None
        self.labels = None
        self.data = None
        self.load()
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.zeros(self.maxLength).type(torch.LongTensor)
        features = self.data[idx] + 1 # adds 1 to keep 0 as padding idx
        nFeatures = len(features)
        if self.train and self.nFeaturesRange is not None and features.size != 0:
            low, high = self.nFeaturesRange
            nFeatures = np.random.randint(min(low,nFeatures), min(high,nFeatures+1))
            features = np.random.choice(features,size=nFeatures,replace=False)
        if nFeatures != 0:
            x[:nFeatures] = torch.LongTensor(features)

        y = self.labels[idx]
        return x, y

    def load(self):
        with open(self.path, 'r', encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')

            if not self.isHashingTrick:
                for row in reader:
                    self.fit(' '.join(row[1:]))
                f.seek(0)

            self.labels = torch.LongTensor([int(row[0]) - 1 for row in reader])
            f.seek(0)

            self.data = [np.array(list(self.to_ids(' '.join(row[1:])))) for row in reader]
         
        self.maxLength = len(max(self.data,key=len))

    """
    def load(self):
        with open(self.path, 'r', encoding="utf-8") as f:
            sparseCounts = self.vectorizer.fit_transform(f)
            f.seek(0)
            reader = csv.reader(f, delimiter=',', quotechar='"')
            labels = [int(row[0]) - 1 for row in reader]
            self.labels = torch.LongTensor(labels)

        # converts to repetition of indices
        self.data = [np.repeat(row.indices,np.abs(row.data).astype(int)) for row in sparseCounts]
        self.maxLength = len(max(self.data,key=len))
    """

class AgNews(CrepeDataset):
    r"""`AG's News <http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html>` dataset.

        The AG's news topic classification dataset is constructed by choosing 4 largest classes from the original corpus. 
        Each class contains 30,000 training samples and 1,900 testing samples. The total number of training samples is 120,000 and testing 7,600.
        classes = {'World': 0, 'Sports': 1, 'Business': 2, 'Sci/Tech': 3}.
        
        Args:
            classes (dict,optional): map class label to label in the data.
            id (string): Id that will be used to refer to this datset.
            file (string,optional): Filename to append to the path. If none then will use ``train.csv`` or ``test.csv``.
            train (bool,optional): If True, creates dataset from ``train.csv`` else ``test.csv``. 
                If false has to give the train `CrepeDataset`.
            isHashingTrick (bool,optional): Whether to use the `hashing trick` rather than a dictionnary.
            nFeaturesRange (tuple,optional): (minFeatures,maxFeatures) If given, each phrase will have a 
                random number of features K extracted. WHere k in range [minFeatures,maxFeatures[. 
                This can be seen as dropout. Only when training.
            trainVocab (Vocabulary,optional): Vocabulary trained on the train set. Mandatory if not `isHashingTrick` and not `train`.
            seed (int, optional): sets the seed for generating random numbers.
            **kwargs: Additional arguments to the vectorizer : `hashing_trick` or `Vocabulary`. 
                Default `hashing_trick`: n=None, hash_function=None, filters=string.punctuation, lower=True,
                                         rmSingleChar=True, split=' ', maxLength=None, mask_zero=True, ngramRange=(1,2).
                Default `Vocabulary`: n=None, hash_function=None, filters=string.punctuation, lower=True,
                                      rmSingleChar=True, split=' ', maxLength=None, mask_zero=True, ngramRange=(1,2).
        """
    def __init__(self,
                 classes={'World': 0, 'Sports': 1, 'Business': 2, 'Sci/Tech': 3},
                 id="ag",
                 **kwargs):
        self.id = id
        self.rootPath = get_path(self.id)
        super(AgNews,self).__init__(self.rootPath,**kwargs)
        self.classes = classes

class AmazonReviewPolarity(CrepeDataset):
    r"""`Amazon Review Polarity Dataset` dataset.

        The Amazon reviews polarity dataset is constructed by taking review score 1 and 2 as negative, and 4 and 5 as positive.
         Samples of score 3 is ignored. In the dataset, class 1 is the negative and class 2 is the positive. 
         Each class has 1,800,000 training samples and 200,000 testing samples.
        
        Args:
            id (string): Id that will be used to refer to this datset.
            file (string,optional): Filename to append to the path. If none then will use ``train.csv`` or ``test.csv``.
            train (bool,optional): If True, creates dataset from ``train.csv`` else ``test.csv``. 
                If false has to give the train `CrepeDataset`.
            isHashingTrick (bool,optional): Whether to use the `hashing trick` rather than a dictionnary.
            nFeaturesRange (tuple,optional): (minFeatures,maxFeatures) If given, each phrase will have a 
                random number of features K extracted. WHere k in range [minFeatures,maxFeatures[. 
                This can be seen as dropout. Only when training.
            trainVocab (Vocabulary,optional): Vocabulary trained on the train set. Mandatory if not `isHashingTrick` and not `train`.
            seed (int, optional): sets the seed for generating random numbers.
            **kwargs: Additional arguments to the vectorizer : `hashing_trick` or `Vocabulary`. 
                Default `hashing_trick`: n=None, hash_function=None, filters=string.punctuation, lower=True,
                                         rmSingleChar=True, split=' ', maxLength=None, mask_zero=True, ngramRange=(1,2).
                Default `Vocabulary`: n=None, hash_function=None, filters=string.punctuation, lower=True,
                                      rmSingleChar=True, split=' ', maxLength=None, mask_zero=True, ngramRange=(1,2).
        """
    def __init__(self,
                 id="amazon",
                 classes={'Positive': 2, 'Negative': 1},
                 **kwargs):
        self.id = id
        self.rootPath = get_path(self.id)
        super(AmazonReviewPolarity,self).__init__(self.rootPath,**kwargs)
        self.classes = classes

class DbPedia(CrepeDataset):
    r"""`DBPedia Ontology Classification Dataset` dataset.

        The DBpedia ontology classification dataset is constructed by picking 14 non-overlapping classes from DBpedia 2014. 
        They are listed in classes.txt. From each of thse 14 ontology classes, we randomly choose 40,000 training samples 
        and 5,000 testing samples. Therefore, the total size of the training dataset is 560,000 and testing dataset 70,000.
        classes = {'Company': 0, 'EducationalInstitution': 1, 'Artist': 2, 'Athlete': 3, 'OfficeHolder': 4, 'MeanOfTransportation': 5, 
                  'Building': 6, 'NaturalPlace': 7, 'Village': 8, 'Animal': 9, 'Plant': 10, 'Album': 11, 'Film': 12, 'WrittenWork': 13}.
        
        Args:
            classes (dict,optional): map class label to label in the data.
            id (string): Id that will be used to refer to this datset.
            file (string,optional): Filename to append to the path. If none then will use ``train.csv`` or ``test.csv``.
            train (bool,optional): If True, creates dataset from ``train.csv`` else ``test.csv``. 
                If false has to give the train `CrepeDataset`.
            isHashingTrick (bool,optional): Whether to use the `hashing trick` rather than a dictionnary.
            nFeaturesRange (tuple,optional): (minFeatures,maxFeatures) If given, each phrase will have a 
                random number of features K extracted. WHere k in range [minFeatures,maxFeatures[. 
                This can be seen as dropout. Only when training.
            trainVocab (Vocabulary,optional): Vocabulary trained on the train set. Mandatory if not `isHashingTrick` and not `train`.
            seed (int, optional): sets the seed for generating random numbers.
            **kwargs: Additional arguments to the vectorizer : `hashing_trick` or `Vocabulary`. 
                Default `hashing_trick`: n=None, hash_function=None, filters=string.punctuation, lower=True,
                                         rmSingleChar=True, split=' ', maxLength=None, mask_zero=True, ngramRange=(1,2).
                Default `Vocabulary`: n=None, hash_function=None, filters=string.punctuation, lower=True,
                                      rmSingleChar=True, split=' ', maxLength=None, mask_zero=True, ngramRange=(1,2).
        """
    def __init__(self,
                 classes = {'Company': 0, 'EducationalInstitution': 1, 'Artist': 2, 'Athlete': 3, 'OfficeHolder': 4, 'MeanOfTransportation': 5, 
                            'Building': 6, 'NaturalPlace': 7, 'Village': 8, 'Animal': 9, 'Plant': 10, 'Album': 11, 'Film': 12, 'WrittenWork': 13},
                 id="dbpedia",
                 **kwargs):
        self.id = id
        self.rootPath = get_path(self.id)
        super(DbPedia,self).__init__(self.rootPath,**kwargs)
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
            id (string): Id that will be used to refer to this datset.
            file (string,optional): Filename to append to the path. If none then will use ``train.csv`` or ``test.csv``.
            train (bool,optional): If True, creates dataset from ``train.csv`` else ``test.csv``. 
                If false has to give the train `CrepeDataset`.
            isHashingTrick (bool,optional): Whether to use the `hashing trick` rather than a dictionnary.
            nFeaturesRange (tuple,optional): (minFeatures,maxFeatures) If given, each phrase will have a 
                random number of features K extracted. WHere k in range [minFeatures,maxFeatures[. 
                This can be seen as dropout. Only when training.
            trainVocab (Vocabulary,optional): Vocabulary trained on the train set. Mandatory if not `isHashingTrick` and not `train`.
            seed (int, optional): sets the seed for generating random numbers.
            **kwargs: Additional arguments to the vectorizer : `hashing_trick` or `Vocabulary`. 
                Default `hashing_trick`: n=None, hash_function=None, filters=string.punctuation, lower=True,
                                         rmSingleChar=True, split=' ', maxLength=None, mask_zero=True, ngramRange=(1,2).
                Default `Vocabulary`: n=None, hash_function=None, filters=string.punctuation, lower=True,
                                      rmSingleChar=True, split=' ', maxLength=None, mask_zero=True, ngramRange=(1,2).
        """
    def __init__(self,
                 classes = {'sports': 0, 'finance': 1, 'entertainment': 2, 'automobile': 3, 'technology': 4},
                 id="sogou",
                 **kwargs):
        self.id = id
        self.rootPath = get_path(self.id)
        super(SogouNews,self).__init__(self.rootPath,**kwargs)
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
            id (string): Id that will be used to refer to this datset.
            file (string,optional): Filename to append to the path. If none then will use ``train.csv`` or ``test.csv``.
            train (bool,optional): If True, creates dataset from ``train.csv`` else ``test.csv``. 
                If false has to give the train `CrepeDataset`.
            isHashingTrick (bool,optional): Whether to use the `hashing trick` rather than a dictionnary.
            nFeaturesRange (tuple,optional): (minFeatures,maxFeatures) If given, each phrase will have a 
                random number of features K extracted. WHere k in range [minFeatures,maxFeatures[. 
                This can be seen as dropout. Only when training.
            trainVocab (Vocabulary,optional): Vocabulary trained on the train set. Mandatory if not `isHashingTrick` and not `train`.
            seed (int, optional): sets the seed for generating random numbers.
            **kwargs: Additional arguments to the vectorizer : `hashing_trick` or `Vocabulary`. 
                Default `hashing_trick`: n=None, hash_function=None, filters=string.punctuation, lower=True,
                                         rmSingleChar=True, split=' ', maxLength=None, mask_zero=True, ngramRange=(1,2).
                Default `Vocabulary`: n=None, hash_function=None, filters=string.punctuation, lower=True,
                                      rmSingleChar=True, split=' ', maxLength=None, mask_zero=True, ngramRange=(1,2).
        """
    def __init__(self,
                 classes = {'Society & Culture': 0, 'Science & Mathematics': 1, 'Health': 2, 'Education & Reference': 3, 
                            'Computers & Internet': 4, 'Sports': 5, 'Business & Finance': 6, 'Entertainment & Music': 7, 
                            'Family & Relationships': 8, 'Politics & Government': 9},
                 id="yahoo",
                 **kwargs):
        self.id = id
        self.rootPath = get_path(self.id)
        super(YahooAnswers,self).__init__(self.rootPath,**kwargs)
        self.classes = classes

class YelpReview(CrepeDataset):
    r"""`Yelp Review Full Star Dataset` dataset.

        The Yelp reviews full star dataset is constructed by randomly taking 130,000 training samples and 10,000 testing samples for each
         review star from 1 to 5. In total there are 650,000 trainig samples and 50,000 testing samples.

        Args:
            id (string): Id that will be used to refer to this datset.
            file (string,optional): Filename to append to the path. If none then will use ``train.csv`` or ``test.csv``.
            train (bool,optional): If True, creates dataset from ``train.csv`` else ``test.csv``. 
                If false has to give the train `CrepeDataset`.
            isHashingTrick (bool,optional): Whether to use the `hashing trick` rather than a dictionnary.
            nFeaturesRange (tuple,optional): (minFeatures,maxFeatures) If given, each phrase will have a 
                random number of features K extracted. WHere k in range [minFeatures,maxFeatures[. 
                This can be seen as dropout. Only when training.
            trainVocab (Vocabulary,optional): Vocabulary trained on the train set. Mandatory if not `isHashingTrick` and not `train`.
            seed (int, optional): sets the seed for generating random numbers.
            **kwargs: Additional arguments to the vectorizer : `hashing_trick` or `Vocabulary`. 
                Default `hashing_trick`: n=None, hash_function=None, filters=string.punctuation, lower=True,
                                         rmSingleChar=True, split=' ', maxLength=None, mask_zero=True, ngramRange=(1,2).
                Default `Vocabulary`: n=None, hash_function=None, filters=string.punctuation, lower=True,
                                      rmSingleChar=True, split=' ', maxLength=None, mask_zero=True, ngramRange=(1,2).
        """
    def __init__(self,
                classes = {'1 star': 1, '2 stars': 2, '3 stars': 3, '4 stars': 4, '5 stars': 5},
                 id="yelp",
                 **kwargs):
        self.id = id
        self.rootPath = get_path(self.id)
        super(YelpReview,self).__init__(self.rootPath,**kwargs)
        self.classes = classes

class YelpReviewPolarity(CrepeDataset):
    r"""`Yelp Review Full Star Dataset` dataset.

        The Yelp reviews polarity dataset is constructed by considering stars 1 and 2 negative, and 3 and 4 positive. For each polarity
        280,000 training samples and 19,000 testing samples are take randomly. In total there are 560,000 trainig samples and 38,000 
        testing samples. Negative polarity is class 1, and positive class 2.

        Args:
            id (string): Id that will be used to refer to this datset.
            file (string,optional): Filename to append to the path. If none then will use ``train.csv`` or ``test.csv``.
            train (bool,optional): If True, creates dataset from ``train.csv`` else ``test.csv``. 
                If false has to give the train `CrepeDataset`.
            isHashingTrick (bool,optional): Whether to use the `hashing trick` rather than a dictionnary.
            nFeaturesRange (tuple,optional): (minFeatures,maxFeatures) If given, each phrase will have a 
                random number of features K extracted. WHere k in range [minFeatures,maxFeatures[. 
                This can be seen as dropout. Only when training.
            trainVocab (Vocabulary,optional): Vocabulary trained on the train set. Mandatory if not `isHashingTrick` and not `train`.
            seed (int, optional): sets the seed for generating random numbers.
            **kwargs: Additional arguments to the vectorizer : `hashing_trick` or `Vocabulary`. 
                Default `hashing_trick`: n=None, hash_function=None, filters=string.punctuation, lower=True,
                                         rmSingleChar=True, split=' ', maxLength=None, mask_zero=True, ngramRange=(1,2).
                Default `Vocabulary`: n=None, hash_function=None, filters=string.punctuation, lower=True,
                                      rmSingleChar=True, split=' ', maxLength=None, mask_zero=True, ngramRange=(1,2).
        """
    def __init__(self,
                 classes={'Positive': 2, 'Negative': 1},
                 id="yelp-polarity",
                 **kwargs):
        self.id = id
        self.rootPath = get_path(self.id)
        super(YelpReviewPolarity,self).__init__(self.rootPath,**kwargs)
        self.classes = classes

def get_dataset(identifier):
    r"""Returns the correct CrepeDataset based on Id in `{ag,amazon,dbpedia,sogou,yahoo,yelp,yelp-polarity}`"""
    if identifier == "ag":
        dataset = AgNews
    elif identifier == "amazon":
        dataset = AmazonReviewPolarity
    elif identifier == "dbpedia":
        dataset = DbPedia
    elif identifier == "sogou":
        dataset = SogouNews
    elif identifier == "yahoo":
        dataset = YahooAnswers
    elif identifier == "yelp":
        dataset = YelpReview 
    elif identifier == "yelp-polarity":
        dataset = YelpReviewPolarity
    else:
        raise ValueError("Unkown dataset identifier: {}".format(identifier))
        
    return dataset

def get_path(identifier):
    r"""Returns the correct root directory to dataset based on Id in `{ag,amazon,dbpedia,sogou,yahoo,yelp,yelp-polarity}`"""
    if identifier == "ag":
        path = "../../data/ag_news_csv"
    elif identifier == "amazon":
        path = "../../data/amazon_review_polarity_csv"
    elif identifier == "dbpedia":
        path = "../../data/dbpedia_csv"
    elif identifier == "sogou":
        path = "../../data/sogou_news_csv"
    elif identifier == "yahoo":
        path = "../../data/yahoo_answers_csv"
    elif identifier == "yelp":
        path = "../../data/yelp_review_full_csv"
    elif identifier == "yelp-polarity":
        path = "../../data/yelp_review_polarity_csv"
    else:
        raise ValueError("Unkown dataset identifier: {}".format(identifier))
        
    return os.path.abspath(os.path.join(os.path.dirname(__file__),path))