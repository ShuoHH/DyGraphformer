import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from dygraph_models.dygraph_encoder import Encoder
from dygraph_models.cross_decoder import Decoder
from dygraph_models.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from dygraph_models.dygraph_embed import DSW_embedding
from dygraph_models.DDGCRN import DGCRM
from math import ceil

class DyGraphformer(nn.Module):
    def __init__(self, data_dim, in_len, out_len, seg_len,batch, win_size = 4,
                d_model=512, d_ff = 1024,gcn_dim=8, n_heads=8, e_layers=3,
                dropout=0.0, gcn_heads=2,gcn_dropout=0.2,baseline = False, device=torch.device('cuda:0')):
        super(DyGraphformer, self).__init__()

        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.batch=batch
        self.merge_win = win_size
        self.out_seg_num=ceil(1.0 * out_len / seg_len)
        self.baseline = baseline
        self.gcn_heads=gcn_heads
        self.gcn_dropout=gcn_dropout
        self.device = device

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, 1, (self.pad_in_len // seg_len), d_model))
        self.enc_node_emb=nn.Parameter(torch.randn(size=(1,data_dim,1,d_model)))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff,gcn_dim,gcn_heads,gcn_dropout, block_depth = 1,dropout = dropout,device=self.device,batch=self.batch, \
                                    out_seg_num=self.out_seg_num,in_seg_num = (self.pad_in_len // seg_len),\
                               ts_dim=self.data_dim,seg_len=self.seg_len)
        

        
    def forward(self, x_seq):
        '''
        x 的输入本来是B T N   DWS之后变成 B N L D
        '''

        if (self.baseline):
            base = x_seq.mean(dim = 1, keepdim = True)
        else:
            base = 0
        batch_size = x_seq.shape[0]
        #实验增加长度
        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim = 1)



        x_seq = self.enc_value_embedding(x_seq)

        #x_seq的维度是B N L D  +1 N L D  -》B N L D
        x_seq += self.enc_pos_embedding
        b,_,l,_=x_seq.shape
        node_emb=self.enc_node_emb.expand(b,-1,l,-1)

        x_seq += node_emb

        x_seq = self.pre_norm(x_seq)
        # # print(x_seq[0, 0, 0, 0])

        enc_out = self.encoder(x_seq)


        return enc_out



if __name__ == '__main__':
    pass


