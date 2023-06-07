import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=32, nc=10):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv1d(nc, 16, 3, 1, 1, bias=False),              # B,  128, 32, 32
            View((-1, 100)),
            nn.BatchNorm1d(100),
            View((-1, 16, 100)),
            nn.ReLU(True),
            nn.Conv1d(16, 32, 3, 1, 1, bias=False),             # B,  256, 16, 16
            View((-1, 100)),
            nn.BatchNorm1d(100),
            View((-1, 32, 100)),
            nn.ReLU(True),
            nn.Conv1d(32, 64, 3, 1, 1, bias=False),             # B,  512,  8,  8
            View((-1, 100)),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            View((-1, 100*64)),                                 # B, 1024*4*4
        )
        # self.conv1 = nn.Conv1d(nc, 4, 3, 1, 1, bias=False)
        # self.batchnorm1 = nn.BatchNorm1d(100)
        # self.relu = nn.ReLU(True)
        # self.conv2 = nn.Conv1d(4, 16, 3, 1, 1, bias=False)
        self.fc_mu = nn.Linear(100*64, z_dim)                            # B, z_dim
        self.fc_logvar = nn.Linear(100*64, z_dim)                            # B, z_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 100*64),                           # B, 1024*8*8
            View((-1, 64, 100)),                               # B, 1024,  8,  8
            nn.ConvTranspose1d(64, 32, 3, 1, 1, bias=False),   # B,  512, 16, 16
            View((-1, 100)),
            nn.BatchNorm1d(100),
            View((-1, 32, 100)),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 16, 3, 1, 1, bias=False),    # B,  256, 32, 32
            View((-1, 100)),
            nn.BatchNorm1d(100),
            View((-1, 16, 100)),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, nc, 3, 1, 1, bias=False),    # B,  128, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x):
    
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)
        
        return x_recon, z, mu, logvar
    
    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)
    

class AAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=100, embedding_dim=128, n_heads=8, normalization=None, nc=20):
        super(AAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.online_encoder = nn.Sequential(
            nn.Linear(z_dim, embedding_dim),
            PositionalEncoding(embedding_dim),
            Transformer(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=3,
            normalization=normalization
        )
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(z_dim, embedding_dim),
            PositionalEncoding(embedding_dim),
            Transformer(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=3,
            normalization=normalization
        )
        )
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim*nc, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim*nc)
        )
        # self.init_embed = nn.Linear(z_dim, embedding_dim)
        # self.pos = PositionalEncoding(embedding_dim)
        self.depos = DEPositionalEncoding(embedding_dim)
        # self.encoder = Transformer(
        #     n_heads=n_heads,
        #     embed_dim=embedding_dim,
        #     n_layers=3,
        #     normalization=normalization
        # )
        
        # self.conv1 = nn.Conv1d(nc, 4, 3, 1, 1, bias=False)
        # self.batchnorm1 = nn.BatchNorm1d(100)
        # self.relu = nn.ReLU(True)
        # self.conv2 = nn.Conv1d(4, 16, 3, 1, 1, bias=False)
        
        self.decoder = Transformer(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=3,
            normalization=normalization
        )
        self.decoder_fc = nn.Linear(embedding_dim, z_dim)
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)
    
    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)
    
    def forward(self, x1, x2):
        # z = self.init_embed(x)
        # z = self.pos(z)
        z1 = self.online_encoder(x1)
        z2 = self.online_encoder(x2)
        predictions_from_view_1 = self.predictor(z1.view(z1.shape[0], -1))
        predictions_from_view_2 = self.predictor(z2.view(z2.shape[0], -1))
        targets_to_view_2 = self.target_encoder(x1)
        targets_to_view_1 = self.target_encoder(x2)
        targets_to_view_1 = targets_to_view_1.view(targets_to_view_1.shape[0], -1)
        targets_to_view_2 = targets_to_view_2.view(targets_to_view_2.shape[0], -1)
        _x_recon = self.decoder(z1)
        _x_recon = self.depos(_x_recon)
        x_recon = self.decoder_fc(_x_recon)
        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        
        return x_recon, z1, loss.mean()
    



class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10, nc=20):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim*nc, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z.view(z.shape[0], -1))


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


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
    
class DEPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(DEPositionalEncoding, self).__init__()
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
        return x - self.pe[:, :x.size(1)]