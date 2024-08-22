import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import math

class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len

        self.linear=nn.Sequential(
            nn.Linear(seg_len, 64),
            nn.Linear(64, d_model)
        )
        self.nom1=nn.LayerNorm(d_model)


    def forward(self, x):

        batch, ts_len, ts_dim = x.shape

        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len).float()
        # print('元',x_segment[0,0])

        x_embed = self.linear(x_segment)
        x_embed=self.nom1(x_embed)


        # print('线性',x_embed[0,0])



        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch, d = ts_dim)
        
        return x_embed