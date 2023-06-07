import torch
import numpy as np
from torch import nn
import math
import torch.nn.functional as F

from reformer_pytorch import LSHSelfAttention
from reformer_pytorch import LSHAttention

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
import time

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
        self.dropout = nn.Dropout(0.5)

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, W_query, W_key, W_val, W_out, h=None, mask=None):
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
        Q = torch.matmul(qflat, W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, W_key).view(shp)
        V = torch.matmul(hflat, W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)
        # attn = self.dropout(attn)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)

        return out, attn


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d,
            'layer': nn.LayerNorm
        }.get(normalization, None)
        self.normalizer = normalizer_class(embed_dim)
        self.embed_dim = embed_dim

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, weights, bias):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return F.batch_norm(input.view(-1, input.size(-1)), running_mean=None, running_var=None, weight=weights, bias=bias, training=True).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return F.instance_norm(input.permute(0, 2, 1), running_mean=None, running_var=None, weight=weights, bias=bias, training=True).permute(0, 2, 1)
        elif isinstance(self.normalizer, nn.LayerNorm):
            return F.layer_norm(input, (self.embed_dim,), weights, bias)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, query_weight, query_bias, key_weight, key_bias, value_weight, value_bias, out_weight, out_bias, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        # queries = self.query_projection(queries).view(B, L, H, -1)
        # keys = self.key_projection(keys).view(B, S, H, -1)
        # values = self.value_projection(values).view(B, S, H, -1)
        queries = F.linear(queries, query_weight, query_bias).view(B, L, H, -1)
        keys = F.linear(keys, key_weight, key_bias).view(B, S, H, -1)
        values = F.linear(values, value_weight, value_bias).view(B, S, H, -1)
        start_time = time.time()
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        print(time.time()-start_time)
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)
        # out = self.out_projection(out)
        out = F.linear(out, out_weight, out_bias)

        return out


# class MultiHeadAttentionLayer(nn.Module):

#     def __init__(
#             self,
#             n_heads,
#             embed_dim,
#             feed_forward_hidden=512,
#             normalization='batch',
#     ):
#         super(MultiHeadAttentionLayer, self).__init__()
#         self.attn = MultiHeadAttention(
#                 n_heads,
#                 input_dim=embed_dim,
#                 embed_dim=embed_dim
#                 )
        
#         self.norm1 = Normalization(embed_dim, normalization)
#         self.ff = nn.Sequential(
#                 nn.Linear(embed_dim, feed_forward_hidden),
#                 nn.ReLU(),
#                 nn.Linear(feed_forward_hidden, embed_dim)
#                 ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
    
#         self.norm2 = Normalization(embed_dim, normalization)
    
#     def forward(self, x):
#         x = self.attn(x)
#         x = self.norm1(x)
#         x = self.ff(x)
#         x = self.norm2(x)
#         return x

#     def functional_forward(self, h, params):
        
        # h = h + self.attn(h, params[f'embedder.layer1_multi.W_query'], params[f'embedder.layer1_multi.W_key'], params[f'embedder.layer1_multi.W_val'], params[f'embedder.layer1_multi.W_out'])
        # h = self.layer1_norm1(h, params[f'embedder.layer1_norm1.normalizer.weight'], params[f'embedder.layer1_norm1.normalizer.bias'])
        # if self.feed_forward_hidden > 0:
        #     h = h + SeqFunction(h, params[f'embedder.layer1_seq.0.weight'], params[f'embedder.layer1_seq.0.bias'], params[f'embedder.layer1_seq.2.weight'], params[f'embedder.layer1_seq.2.bias'], self.feed_forward_hidden)
        # else:
        #     h = h + SeqFunction(h, params[f'embedder.layer1_seq.0.weight'], params[f'embedder.layer1_seq.0.bias'], self.feed_forward_hidden)
        # h = self.layer1_norm1(h, params[f'embedder.layer1_norm2.normalizer.weight'], params[f'embedder.layer1_norm2.normalizer.bias']) 


#### positional encoding ####
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
        return self.dropout(x + self.pe[:, :x.size(1)])
    

class MultiHeadAttentionLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__()

        self.embed_dim = embed_dim
        self.self_attn = MultiHeadAttention(
                n_heads,
                input_dim=embed_dim,
                embed_dim=embed_dim
                )
        
        self.dropout = nn.Dropout(0.5)
        self.norm1 = Normalization(embed_dim, normalization)
        # self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
                nn.Linear(embed_dim, feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
    
        # self.norm2 = nn.LayerNorm(embed_dim)
        self.norm2 = Normalization(embed_dim, normalization)

        self.feed_forward_hidden = feed_forward_hidden
    
    def forward(self, x):
        x, attn = self.self_attn(x)
        x = x+ self.dropout(x)
        x = self.norm1(x)
        x = self.ff(x)
        x = self.norm2(x)
        return x

    def functional_forward(self, x, params, layer_num):

        # x2 = self.self_attn(x, params[f'embedder.layer{layer_num}.self_attn.W_query'], params[f'embedder.layer{layer_num}.self_attn.W_key'], params[f'embedder.layer{layer_num}.self_attn.W_val'], params[f'embedder.layer{layer_num}.self_attn.W_out'])
        x, attn = self.self_attn(x, params[f'embedder.layer{layer_num}.self_attn.W_query'], params[f'embedder.layer{layer_num}.self_attn.W_key'], params[f'embedder.layer{layer_num}.self_attn.W_val'], params[f'embedder.layer{layer_num}.self_attn.W_out'])
        x = x + self.dropout(x)
        x = self.norm1(x, params[f'embedder.layer{layer_num}.norm1.normalizer.weight'], params[f'embedder.layer{layer_num}.norm1.normalizer.bias'])
        # x = F.layer_norm(x, (self.embed_dim,), params[f'embedder.layer{layer_num}.norm1.weight'], params[f'embedder.layer{layer_num}.norm1.bias'])

        x = x + SeqFunction(x, params[f'embedder.layer{layer_num}.ff.0.weight'], params[f'embedder.layer{layer_num}.ff.0.bias'], params[f'embedder.layer{layer_num}.ff.2.weight'], params[f'embedder.layer{layer_num}.ff.2.bias'], self.feed_forward_hidden)
        x = x + self.dropout(x)
        # x = F.layer_norm(x, (self.embed_dim,), params[f'embedder.layer{layer_num}.norm2.weight'], params[f'embedder.layer{layer_num}.norm2.bias'])
        x = self.norm2(x, params[f'embedder.layer{layer_num}.norm2.normalizer.weight'], params[f'embedder.layer{layer_num}.norm2.normalizer.bias'])
        
        return x, attn

        
        
def SeqFunction(input, weight1, bias1, weight2, bias2, feed_forward_hidden):
    if feed_forward_hidden > 0:
        x = F.linear(input, weight1, bias1)
        x = F.relu(x)
        x = F.linear(x, weight2, bias2)
    else:
        x = F.linear(input, weight1, bias1)
    
    return x



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
        # self.layers = nn.Sequential(*(
        #     MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
        #     for _ in range(n_layers)
        # ))

        self.layer1 = MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
        self.layer2 = MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
        self.layer3 = MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)

        '''
        self.layer1_multi = MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
        self.layer1_norm1 = Normalization(embed_dim, normalization)
        self.layer1_seq = nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
        self.layer1_norm2 = Normalization(embed_dim, normalization)

        self.layer2_multi = MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
        self.layer2_norm1 = Normalization(embed_dim, normalization)
        self.layer2_seq = nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
        self.layer2_norm2 = Normalization(embed_dim, normalization)

        self.layer3_multi = MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
        self.layer3_norm1 = Normalization(embed_dim, normalization)
        self.layer3_seq = nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
        self.layer3_norm2 = Normalization(embed_dim, normalization)
        '''



    def forward(self, x):

        h = x

        h, attn1 = self.layer1(h)
        h, attn2 = self.layer2(h)
        h, attn3 = self.layer3(h)

        return h, attn3

    # def functional_forward(self, x, params, mask=None):

    #     assert mask is None, "TODO mask not yet supported!"

    #     # Batch multiply to get initial embeddings of nodes
    #     h = F.linear(x.view(-1, x.size(-1)), params[f'embedder.init_embed.weight'], params[f'embedder.init_embed.bias']).view(*x.size()[:2], -1) if self.init_embed is not None else x

    #     h = h + self.layer1_multi(h, params[f'embedder.layer1_multi.W_query'], params[f'embedder.layer1_multi.W_key'], params[f'embedder.layer1_multi.W_val'], params[f'embedder.layer1_multi.W_out'])
    #     h = self.layer1_norm1(h, params[f'embedder.layer1_norm1.normalizer.weight'], params[f'embedder.layer1_norm1.normalizer.bias'])
    #     if self.feed_forward_hidden > 0:
    #         h = h + SeqFunction(h, params[f'embedder.layer1_seq.0.weight'], params[f'embedder.layer1_seq.0.bias'], params[f'embedder.layer1_seq.2.weight'], params[f'embedder.layer1_seq.2.bias'], self.feed_forward_hidden)
    #     else:
    #         h = h + SeqFunction(h, params[f'embedder.layer1_seq.0.weight'], params[f'embedder.layer1_seq.0.bias'], self.feed_forward_hidden)
    #     h = self.layer1_norm1(h, params[f'embedder.layer1_norm2.normalizer.weight'], params[f'embedder.layer1_norm2.normalizer.bias'])


    #     h = h + self.layer2_multi(h, params[f'embedder.layer2_multi.W_query'], params[f'embedder.layer2_multi.W_key'], params[f'embedder.layer2_multi.W_val'], params[f'embedder.layer2_multi.W_out'])
    #     h = self.layer2_norm1(h, params[f'embedder.layer2_norm1.normalizer.weight'], params[f'embedder.layer2_norm1.normalizer.bias'])
    #     if self.feed_forward_hidden > 0:
    #         h = h + SeqFunction(h, params[f'embedder.layer2_seq.0.weight'], params[f'embedder.layer2_seq.0.bias'], params[f'embedder.layer2_seq.2.weight'], params[f'embedder.layer2_seq.2.bias'], self.feed_forward_hidden)
    #     else:
    #         h = h + SeqFunction(h, params[f'embedder.layer2_seq.0.weight'], params[f'embedder.layer2_seq.0.bias'], self.feed_forward_hidden)
    #     h = self.layer2_norm1(h, params[f'embedder.layer2_norm2.normalizer.weight'], params[f'embedder.layer2_norm2.normalizer.bias'])


    #     h = h + self.layer3_multi(h, params[f'embedder.layer3_multi.W_query'], params[f'embedder.layer3_multi.W_key'], params[f'embedder.layer3_multi.W_val'], params[f'embedder.layer3_multi.W_out'])
    #     h = self.layer3_norm1(h, params[f'embedder.layer3_norm1.normalizer.weight'], params[f'embedder.layer3_norm1.normalizer.bias'])
    #     if self.feed_forward_hidden > 0:
    #         h = h + SeqFunction(h, params[f'embedder.layer3_seq.0.weight'], params[f'embedder.layer3_seq.0.bias'], params[f'embedder.layer3_seq.2.weight'], params[f'embedder.layer3_seq.2.bias'], self.feed_forward_hidden)
    #     else:
    #         h = h + SeqFunction(h, params[f'embedder.layer3_seq.0.weight'], params[f'embedder.layer3_seq.0.bias'], self.feed_forward_hidden)
    #     h = self.layer3_norm1(h, params[f'embedder.layer3_norm2.normalizer.weight'], params[f'embedder.layer3_norm2.normalizer.bias'])

    #     return (
    #         h,  # (batch_size, graph_size, embed_dim)
    #         h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
    #     )
    
    def functional_forward(self, x, params):


        # Batch multiply to get initial embeddings of nodes
        h = x

        h, attn1 = self.layer1.functional_forward(h, params, layer_num = 1)
        h, attn2 = self.layer2.functional_forward(h, params, layer_num = 2)
        h, attn3 = self.layer3.functional_forward(h, params, layer_num = 3)

        return h, attn3 # (batch_size, graph_size, embed_dim)
           
    
class MultiHeadAttentionLayer1(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer1, self).__init__()

        self.embed_dim = embed_dim
        self.self_attn = MultiHeadAttention(
                n_heads,
                input_dim=embed_dim,
                embed_dim=embed_dim
                )
        
        self.dropout = nn.Dropout(0.5)
        self.norm1 = Normalization(embed_dim, normalization)
        # self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
                nn.Linear(embed_dim, feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
    
        # self.norm2 = nn.LayerNorm(embed_dim)
        self.norm2 = Normalization(embed_dim, normalization)

        self.feed_forward_hidden = feed_forward_hidden
    
    def forward(self, x):
        x = x+ self.dropout(self.self_attn(x))
        x = self.norm1(x)
        x = self.ff(x)
        x = self.norm2(x)
        return x

    def functional_forward(self, x, params, layer_num):

        # x2 = self.self_attn(x, params[f'embedder.layer{layer_num}.self_attn.W_query'], params[f'embedder.layer{layer_num}.self_attn.W_key'], params[f'embedder.layer{layer_num}.self_attn.W_val'], params[f'embedder.layer{layer_num}.self_attn.W_out'])
        x = x + self.dropout(self.self_attn(x, params[f'embedder.layer{layer_num}.self_attn.W_query'], params[f'embedder.layer{layer_num}.self_attn.W_key'], params[f'embedder.layer{layer_num}.self_attn.W_val'], params[f'embedder.layer{layer_num}.self_attn.W_out']))
        x = self.norm1(x, params[f'embedder.layer{layer_num}.norm1.normalizer.weight'], params[f'embedder.layer{layer_num}.norm1.normalizer.bias'])
        # x = F.layer_norm(x, (self.embed_dim,), params[f'embedder.layer{layer_num}.norm1.weight'], params[f'embedder.layer{layer_num}.norm1.bias'])

        x = x + SeqFunction(x, params[f'embedder.layer{layer_num}.ff.0.weight'], params[f'embedder.layer{layer_num}.ff.0.bias'], params[f'embedder.layer{layer_num}.ff.2.weight'], params[f'embedder.layer{layer_num}.ff.2.bias'], self.feed_forward_hidden)
        x = x + self.dropout(x)
        # x = F.layer_norm(x, (self.embed_dim,), params[f'embedder.layer{layer_num}.norm2.weight'], params[f'embedder.layer{layer_num}.norm2.bias'])
        x = self.norm2(x, params[f'embedder.layer{layer_num}.norm2.normalizer.weight'], params[f'embedder.layer{layer_num}.norm2.normalizer.bias'])
        
        return x
