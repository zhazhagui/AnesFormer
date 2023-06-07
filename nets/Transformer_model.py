import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import collections
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def plot_confusion_matrix(cm, labels_name, title):
    ccm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(ccm, interpolation='nearest', cmap=plt.cm.Blues, vmax=1, vmin=0)    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    for first_index in range(len(cm)):  # 第几行
        for second_index in range(len(cm[first_index])):  # 第几列
            temp = cm[first_index][second_index]
            plt.text(second_index, first_index, int(temp), va='center',
                    ha='center',
                    fontsize=13.5)

    plt.xlabel('True label')    
    plt.ylabel('Predicted label')

class Transmodel(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=40, embedding_dim=128, n_heads=8, normalization=None, nc=10):
        super(Transmodel, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.init_embed = nn.Linear(z_dim, embedding_dim)
        self.pos = PositionalEncoding(embedding_dim)
        self.encoder = Transformer(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=3,
            normalization=normalization
        )
        # self.conv1 = nn.Conv1d(nc, 4, 3, 1, 1, bias=False)
        # self.batchnorm1 = nn.BatchNorm1d(100)
        # self.relu = nn.ReLU(True)
        # self.conv2 = nn.Conv1d(4, 16, 3, 1, 1, bias=False)
        
        self.decoder_1 = nn.Linear(embedding_dim*nc, z_dim)
        self.decoder_2 = nn.Linear(z_dim, 2)
    
    def forward(self, x):
        z = self.init_embed(x)
        z = self.pos(z)
        z = self.encoder(z).view(z.shape[0], -1)
        z = self.decoder_1(z)
        z = self.decoder_2(z)
        
        return z

class CNNTransmodel(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=40, embedding_dim=128, n_heads=8, normalization=None, nc=10):
        super(CNNTransmodel, self).__init__()
        self.z_dim = z_dim
        self.nc = nc

        self.conv1 = nn.Conv1d(20, 32, 3, 1, 1)
        # self.batchnorm1 = nn.BatchNorm1d(40, False)
        self.pooling1 = nn.MaxPool1d(2, 2)

        self.conv2 = nn.Conv1d(32, 64, 3, 1, 1)
        # self.batchnorm2 = nn.BatchNorm1d(20, False)
        self.pooling2 = nn.MaxPool1d(2, 2)

        self.init_embed = nn.Linear(10, embedding_dim)
        self.pos = PositionalEncoding(embedding_dim)
        self.encoder = Transformer(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=3,
            normalization=normalization
        )
        # self.conv1 = nn.Conv1d(nc, 4, 3, 1, 1, bias=False)
        # self.batchnorm1 = nn.BatchNorm1d(100)
        # self.relu = nn.ReLU(True)
        # self.conv2 = nn.Conv1d(4, 16, 3, 1, 1, bias=False)
        
        self.decoder_1 = nn.Linear(embedding_dim*64, 10)
        self.decoder_2 = nn.Linear(10, 2)
    
    def forward(self, x):
        x = F.elu(self.conv1(x))
        # x = self.batchnorm1(x.view(-1,40)).view(-1,32,40)
        x = self.pooling1(x)
        # Layer 2
        x = F.elu(self.conv2(x))
        # x = self.batchnorm2(x.view(-1,20)).view(-1,64,20)
        x = self.pooling2(x)
        z = self.init_embed(x)
        z = self.pos(z)
        z = self.encoder(z).view(z.shape[0], -1)
        z = self.decoder_1(z)
        z = self.decoder_2(z)
        
        return z

class Transformer(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(Transformer, self).__init__()

        self.feed_forward_hidden = feed_forward_hidden

        self.n_layers = n_layers
        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x):

        h = x

        h = self.layers(h)

        return h

    
class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)    

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d,
            'layer': nn.LayerNorm
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        elif isinstance(self.normalizer, nn.LayerNorm):
            return self.normalizer(input)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(0.5)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]