import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal

class HashEmbedding(nn.Module):
    r"""Type of embedding which uses multiple hashes to approximate an Embedding layer with more parameters.

    This module is a new Embedding module that compresses the number of parameters. They are a 
    generalization of vanilla Embeddings and the `hashing trick`. For more details, check Svenstrup, 
    Dan Tito, Jonas Hansen, and Ole Winther. "Hash embeddings for efficient word
    representations." Advances in Neural Information Processing Systems. 2017.
    
    For each elements (usually word indices) in the input (mini_batch, sequence_length) the default 
    computations are:

    .. math::

            \begin{array}{ll}
            H_i = E_{D_2^i(D_1(w)))} \ \forall i=1...k\\
            c_w = (H_1(w), ..., H_k(w))^T\\
            p_w = P_{D_1(w)}\\
            \hat{e}_w = p_w \cdot c_w\\
            e_w = \mathrm{concatenate}(\hat{e}_w,p_w)\\
            \end{array}

    where :math:`w:[0,T]` is the element of the input (word index), :math:`D_1:[0,T)\to [0,K)` 
    is the token to ID hash/dictionnary, :math:`D_2:[0,K)\to[0,B)` is the ID to Bucket hash, 
    :math:`E:\mathbb R^{B*d}` is the shared pool of embeddings, :math:`c_w:\mathbb R^{k*d}` contains all
    the vector embeddings to which :math:`w` maps, :math:`e_w:\mathbb R^{d+k}` is the outputed word
    embedding for :math:`w`.

    Args:
        num_embeddings (int): the number of different embeddings. K in the paper.
        embedding_dim (int): the size of each embedding vector in the shared pool. d in the paper.
        num_buckets (int,optional): the size of the shared pool of embeddings. B in the paper. 
            Typically num_buckets * 10 < num_embeddings.
        num_hashes (int,optional): the number of different hash functions. k in the paper. 
            Typically in [1,3].
        train_sharedEmbed (bool,optional): whether to train the shared pool of embeddings E.
        train_weight (bool,optional): whether to train the importance parameters / weight P.
        append_weight (bool,optional): whether to append the importance parameters / weight pw.
        aggregation_mode ({"sum","mean","concatenate"},optional): how to aggregate the component vectors
            of the different hashes.  
        mask_zero (bool, optional): whether the 0 input is a special "padding" value to mask out.
        seed (int, optional): sets the seed for generating random numbers.

    Attributes:
        shared_embeddings (nn.Embedding): the shared pool of embeddings of shape (num_buckets, embedding_dim). 
            E in the paper. 
        importance_weights (nn.Embedding): the importance parameters / weight of shape 
            (num_embeddings, num_hashes). P in the paper. 
        output_dim (int): effective outputed number of embeddings.

    Shape:
        - Input: LongTensor `(N, W)`, N = mini-batch, W = number of indices to extract per mini-batch
        - Output: `(N, W, output_dim)`, output_dim is the effective embedding dim.

    Examples::
        >>> # an HashEmbedding module containing approximating nn.Embedding(10, 5) with less param
        >>> embedding = HashEmbedding(10,5,append_weight=False)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]))
        >>> embedding(input)

        Variable containing:
        (0 ,.,.) = 
        1.00000e-04 *
           0.3988  0.5234 -0.6148  0.3000 -1.5525
           0.1259  0.4142 -0.8613  0.3018 -1.3547
           0.1367  0.2638 -0.2993  0.9541 -1.7194
          -0.4672 -0.7971 -0.2009  0.7829 -0.9448

        (1 ,.,.) = 
        1.00000e-04 *
           0.1367  0.2638 -0.2993  0.9541 -1.7194
          -0.0878 -0.1680  0.3896  0.5288 -0.2060
           0.1259  0.4142 -0.8613  0.3018 -1.3547
          -0.3098  0.0357 -0.7532 -0.1216 -0.0366
        [torch.FloatTensor of size 2x4x5]

        >>> # example with mask_zero which corresponds to padding_idx=0
        >>> embedding = HashEmbedding(10,5,append_weight=False, mask_zero=True)
        >>> input = Variable(torch.LongTensor([[0,2,0,5]]))
        >>> embedding(input)

        Variable containing:
        (0 ,.,.) = 
        1.00000e-04 *
           0.0000  0.0000  0.0000  0.0000  0.0000
          -1.4941 -1.3775 -0.5797  1.3187 -0.0555
           0.0000  0.0000  0.0000  0.0000  0.0000
          -0.7717 -0.5569 -0.1397  1.1101 -0.0939
        [torch.FloatTensor of size 1x4x5]
    """
    def __init__(self, num_embeddings, embedding_dim, num_buckets=None, num_hashes=2, train_sharedEmbed=True,
                 train_weight=True, append_weight=True, aggregation_mode='sum', mask_zero=False,seed=None):
        super(HashEmbedding, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_hashes = num_hashes
        defaultNBuckets = (num_embeddings * self.num_hashes)//(self.embedding_dim)
        self.num_buckets = num_buckets - 1 if num_buckets is not None else defaultNBuckets
        self.train_sharedEmbed = train_sharedEmbed
        self.train_weight = train_weight
        self.append_weight = append_weight
        self.padding_idx = 0 if mask_zero else None
        self.seed = seed
        
        self.importance_weights = nn.Embedding(self.num_embeddings,
                                              self.num_hashes)
        self.shared_embeddings = nn.Embedding(self.num_buckets + 1,
                                            self.embedding_dim,
                                            padding_idx=self.padding_idx)
        self.hashes = None
        
        if aggregation_mode == 'sum':
            self.aggregate = lambda x: torch.sum(x, dim=-1)
        elif aggregation_mode == 'concatenate':
            # little bit quicker than permute/contiguous/view
            self.aggregate = lambda x: torch.cat([x[:,:,:,i] for i in range(self.num_hashes)], dim=-1) 
        elif aggregation_mode == 'mean':
            self.aggregate = lambda x: torch.mean(x, dim=-1)
        else:
            raise ValueError('unknown aggregation function {}'.format(aggregation_mode))
        
        self.output_dim = self.embedding_dim 
        if aggregation_mode == "concatenate":
            self.output_dim *= self.num_hashes
        if self.append_weight:
            self.output_dim += self.num_hashes
            
        self.reset_parameters()   
        
    def reset_parameters(self,
                        init_shared=lambda x: normal(x,std=0.1),
                        init_importance=lambda x: normal(x,std=0.0005)):
        """Resets the trainable parameters."""
        def set_constant_row(parameters,iRow=0,value=0):
            """Return `parameters` with row `iRow` as s constant `value`."""
            data = parameters.data
            data[iRow,:] = value
            return torch.nn.Parameter(data,requires_grad=parameters.requires_grad)

        np.random.seed(self.seed)
        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.shared_embeddings.weight = init_shared(self.shared_embeddings.weight)
        self.importance_weights.weight = init_importance(self.importance_weights.weight)

        if self.padding_idx is not None:
            # Unfortunately has to set weight to 0 even when paddingIdx = 0
            self.shared_embeddings.weight = set_constant_row(self.shared_embeddings.weight)
            self.importance_weights.weight = set_constant_row(self.importance_weights.weight)

        self.shared_embeddings.weight.requires_grad = self.train_sharedEmbed
        self.importance_weights.weight.requires_grad = self.train_weight

        self.hashes = torch.from_numpy((np.random.randint(0, 2 ** 30,
                                                          size=(self.num_embeddings, self.num_hashes)
                                                         ) % self.num_buckets) + 1 
                                      ).type(torch.LongTensor)
        
    def _idx_hash(self, inputs, maxOutput, mask_zero=True):
        r"""Hash function for integers used to map indices of different sizes.
        
        Args:
            inputs (torch.Tensor): indices to hash.
            maxOutput (int): maximum integer to output. I.e size of table to access.
            mask_zero (bool,optional): whether should only map zero input to zero.
            
        To Do:
            Should enable :math:`\hat{D} \neq D_1`.
        """  
        if mask_zero:
            idx_zero = inputs == 0
            # shouldn't map non zero vectors to 0
            inputs = inputs%(maxOutput-1) + 1
            inputs[idx_zero] = 0
            return inputs
        else:
            return inputs%maxOutput
            
    def forward(self, input):  
        idx_hashes = self._idx_hash(input,self.num_embeddings,mask_zero=self.padding_idx is not None)
        idx_importance_weights = self._idx_hash(input,self.num_embeddings,mask_zero=False)
        idx_shared_embeddings = self.hashes[idx_hashes.data,:]
        
        shared_embedding = torch.stack([self.shared_embeddings(idx_shared_embeddings[:,:,iHash]) 
                                        for iHash in range(self.num_hashes)], dim=-1)
        importance_weight = self.importance_weights(idx_importance_weights)
        importance_weight = importance_weight.unsqueeze(-2)
        word_embedding = self.aggregate(importance_weight*shared_embedding)
        if self.append_weight:
            # concateates the vector with the weights
            word_embedding = torch.cat([word_embedding,importance_weight.squeeze(-2)],dim=-1) 
        return word_embedding