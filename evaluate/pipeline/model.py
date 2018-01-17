import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import xavier_uniform, normal, constant

from evaluate.pipeline.embedding import HashEmbedding

def text_embedding(x):
    """Makes a phrase embedding out of all the word embeddings which constitutes it."""
    return torch.sum(x,dim=1)

class ModelNoDict(nn.Module):
    """Simple model that was use in the paper when no dictionnary.

    Args:
        maxWords (int): Maximum number of different words. I.e number of rows in the embedding.
        embeddingDim (int): The size of each embedding vector in the shared pool. d in the paper.
        num_classes (int): The number of different classes to predict from.
        seed (int, optional): sets the seed for generating random numbers.
        isMasking (bool, optional): whether the 0 input is a special "padding" value to mask out.
        isHash (bool,optional): Whether should uhe `HashEmbeddings` rather than `Embeddings`.
        kwargs: Additional arguments to the `HashEmbeddings`.
    """
    def __init__(self,maxWords,embeddingDim,nClasses,seed=3,isMasking=True,isHash=False,**kwargs):
        super().__init__()
        
        self.seed = seed
        self.paddingIdx = 0 if isMasking else None
        self.isHash = isHash
        if self.isHash:
            self.embedding = HashEmbedding(maxWords,embeddingDim,mask_zero=isMasking,seed=seed,**kwargs)
        else:
            self.embedding = nn.Embedding(maxWords,embeddingDim,padding_idx=self.paddingIdx)
        self.text_embedding = text_embedding
        
        self.outDim = self.embedding.output_dim if isHash else embeddingDim
        self.fc1 = nn.Linear(self.outDim, nClasses)
        
        self.reset_parameters()
        
    def reset_parameters(self,
                         init_fc_w=xavier_uniform,
                         init_fc_b=lambda x: constant(x,0),
                         init_embed=lambda x: normal(x,std=0.05),
                         **kwargs):
        """Resets the trainable weights."""
        def set_constant_row(parameters,iRow=0,value=0):
            """Return `parameters` with row `iRow` as s constant `value`."""
            data = parameters.data
            data[iRow,:] = value
            return torch.nn.Parameter(data,requires_grad=parameters.requires_grad)

        np.random.seed(self.seed)
        if self.seed is not None:
            torch.manual_seed(self.seed)

        if not self.isHash:
            self.embedding.weight = init_embed(self.embedding.weight)

            if self.paddingIdx is not None:
                # Unfortunately has to set weight to 0 even when paddingIdx =0
                self.embedding.weight = set_constant_row(self.embedding.weight)
        else:
            self.embedding.reset_parameters(**kwargs)

        self.fc1.weight = init_fc_w(self.fc1.weight)
        self.fc1.biais= init_fc_b(self.fc1.weight)
        
    def forward(self,x):
        x = self.embedding(x)
        x = self.text_embedding(x) 
        x = self.fc1(x)
        return x