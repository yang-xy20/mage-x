import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from onpolicy.algorithms.utils.MAGIC.vit import ViT, Attention, PreNorm, Transformer, CrossAttention, FeedForward
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import random


class Invariant(nn.Module):
    def __init__(self, hidden_dim=128, heads=4, dim_head=32, mlp_dim=128, dropout=0.):
        super().__init__()
        self.attn_net = Transformer(hidden_dim, depth = 1, heads = heads, dim_head = dim_head, mlp_dim = mlp_dim, dropout = 0.)
       
    def forward(self, x):
        B = x.shape[0]
        all = self.attn_net(x)
        all = all.mean(dim=1)
        return all