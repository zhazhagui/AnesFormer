import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches

from nets.graph_encoder_meta import Transformer, PositionalEncoding
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many
import torch.nn.functional as F
from nets.graph_encoder_meta import Normalization
from einops.layers.torch import Rearrange, Reduce



class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )


class AttentionModel(nn.Module):

    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None):
        super(AttentionModel, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        
        self.init_embed = nn.Linear(input_dim, embedding_dim)
        # self.init_embed_2 = nn.Linear(22, embedding_dim)
        # self.conv = nn.Conv1d(400, 100, kernel_size=3, padding = 0 )
        # self.norm1 = Normalization(input_dim, normalization)
        # self.norm2 = Normalization(embedding_dim, normalization)
        # self.conv1 = nn.Conv2d(1, 4, (1, 51), (1, 1))
        # self.batchnorm = nn.BatchNorm2d(4)
        # self.leakyrelu = nn.LeakyReLU(0.2)
        # self.conv2 = nn.Conv2d(4, embedding_dim, (16, 5), stride=(1, 5))
        # self.rearrange = Rearrange('b e (h) (w) -> b (h w) e')
        # self.conv1 = nn.Conv1d(1, 2, kernel_size=3, padding=1, stride=1)
        # self.batchnorm = nn.BatchNorm1d(100)
        # self.pool = nn.MaxPool1d(2,2,0)
        # self.leakyrelu = nn.LeakyReLU(0.2)
        # self.conv2 = nn.Conv1d(2, 4, kernel_size=3, stride=1, padding=1)
        # self.conv1 = nn.Conv1d(10, 16, 3, 1, 1)
        # #self.batchnorm1 = nn.BatchNorm1d(100, False)
        # self.pooling1 = nn.MaxPool1d(2, 2)

        # self.conv2 = nn.Conv1d(16, 32, 3, 1, 1)
        # #self.batchnorm2 = nn.BatchNorm1d(50, False)
        # self.pooling2 = nn.MaxPool1d(2, 2)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.embedder = Transformer(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )
        self.decoder1 = nn.Linear(embedding_dim*20, embedding_dim)  # 这里用全连接层代替了decoder， 其实也可以加一下Transformer的decoder试一下效果
        self.decoder2 = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        # self.dropout = nn.Dropout(0.2)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        # self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)


    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            embeddings, _ = self.embedder(self._init_embed(input))

        _log_p, pi = self._inner(input, embeddings)

        cost, mask = self.problem.get_costs(input, pi)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            return cost, ll, pi

        return cost, ll

    def functional_forward(self, input, params, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        # embeddings = F.elu(F.conv1d(input, params[f'conv1.weight'], params[f'conv1.bias'], stride=1, padding=1))
        # #embeddings = F.batch_norm(embeddings.view(-1, embeddings.size(-1)), running_mean=None, running_var=None, weight=params[f'batchnorm1.weight'], bias=params[f'batchnorm1.bias'], training=True).view(*embeddings.size())
        # embeddings = self.pooling1(embeddings)
        # embeddings = F.elu(F.conv1d(embeddings, params[f'conv2.weight'], params[f'conv2.bias'], stride=1, padding=1))
        # #embeddings = F.batch_norm(embeddings.view(-1, embeddings.size(-1)), running_mean=None, running_var=None, weight=params[f'batchnorm2.weight'], bias=params[f'batchnorm2.bias'], training=True).view(*embeddings.size())
        # embeddings = self.pooling2(embeddings).view(embeddings.shape[0], -1)
        embeddings = F.linear(input, params[f'init_embed.weight'], params[f'init_embed.bias'])
        # embeddings = F.conv1d(input.view(-1,input.size(2),input.size(3)), params[f'conv.weight'], params[f'conv.bias'],).view(-1, 7, 2000)
        # embeddings = self.batchnorm(embeddings, params[f'batchnorm.normalizer.weight'], params[f'batchnorm.normalizer.bias'])
        # embeddings = self.norm1(input, params[f'norm1.normalizer.weight'], params[f'norm1.normalizer.bias'])
        # embeddings = self.dropout(embeddings)
        # embeddings = F.linear(input, params[f'init_embed_1.weight'], params[f'init_embed_1.bias']).view(input.shape[0], input.shape[1], -1)
        # embeddings = self.dropout(embeddings)
        # embeddings = F.linear(input, params[f'init_embed_1.weight'], params[f'init_embed_1.bias'])
        '''
        embeddings = F.conv2d(input, params[f'conv1.weight'], params[f'conv1.bias'])
        embeddings = F.batch_norm(embeddings, running_mean=None, running_var=None, weight=params[f'batchnorm.weight'], bias=params[f'batchnorm.bias'], training=True)
        embeddings = F.leaky_relu(embeddings)
        embeddings = F.conv2d(embeddings, params[f'conv2.weight'], params[f'conv2.bias'], stride=(1,5))
        '''
        # embeddings = F.conv1d(input.view(-1,1,self.input_dim), params[f'conv1.weight'], params[f'conv1.bias'], stride=1, padding=1)
    
        # embeddings = self.pool(embeddings)
        # embeddings = F.leaky_relu(embeddings)
        # embeddings = F.conv1d(embeddings, params[f'conv2.weight'], params[f'conv2.bias'], stride=1, padding=1)
        # embeddings = self.pool(embeddings)
        # # embeddings = self.rearrange(embeddings)
        # embeddings = embeddings.view(-1, 10, self.input_dim)

        embeddings = self.pos_encoder(embeddings)
        embeddings, attn = self.embedder.functional_forward(embeddings, params)
        embeddings = embeddings.view(embeddings.shape[0], -1)
        # assert not torch.isnan(embeddings).any()
        outputs = F.linear(embeddings, params[f'decoder1.weight'], params[f'decoder1.bias'])
        outputs = F.layer_norm(outputs, (self.embedding_dim,), params[f'norm.weight'], params[f'norm.bias'])
        # outputs = self.dropout(outputs)
        outputs = F.linear(outputs, params[f'decoder2.weight'], params[f'decoder2.bias'])
        

        return outputs, embeddings, attn

 