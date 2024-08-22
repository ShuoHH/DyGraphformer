import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from dygraph_models.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from math import ceil
from dygraph_models.DDGCRN import DGCRM
import numpy as np

class SegMerging(nn.Module):
    '''
    Segment Merging Layer.
    The adjacent `win_size' segments in each dimension will be merged into one segment to
    get representation of a coarser scale
    we set win_size = 2 in our paper
    '''
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        """
        x: B, ts_d, L, d_model
        """
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0: 
            pad_num = self.win_size - pad_num   #计算出还有添加一个
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim = -2)   #将最后几个复制补齐

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])  #一次性取出num|size个数 级联  级联数量还是num|size
        x = torch.cat(seg_to_merge, -1)  # [B, ts_d, seg_num/win_size, win_size*d_model]

        x = self.norm(x)
        x = self.linear_trans(x)

        return x

class scale_block(nn.Module):
    '''
    We can use one segment merging layer followed by multiple TSA layers in each scale
    the parameter `depth' determines the number of TSA layers used in each scale
    We set depth = 1 in the paper   本类其实定义了当前尺度层
    '''
    def __init__(self,Batch, win_size, d_model, n_heads, d_ff,gcn_dim,gcn_heads,gcn_dropout, depth, dropout,device, \
                    seg_num,out_seg_num,ts_dim,seg_len ):
        super(scale_block, self).__init__()



        self.linear_pred = nn.Linear(d_model,seg_len)  # 用来最后每一层的输出映射
        self.linear_hid=nn.Linear(seg_num,out_seg_num)
        self.device=device
        if (win_size > 1):
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None
        
        self.encode_layers = nn.ModuleList()
        # self.res_nom=nn.LayerNorm(d_model)



        #定义每一个encoder深度，为encoder实例化一个双层注意力对象 ,每一层都共享
        # depth=2
        for i in range(depth):

            self.encode_layers.append( nn.ModuleList([TwoStageAttentionLayer(seg_num,  d_model, n_heads, d_ff, dropout),
                                       DGCRM(ts_dim=ts_dim,batch=Batch,d_model=d_model,gcn_dim=gcn_dim,gcn_heads=gcn_heads,dropout=gcn_dropout)]))

        #
        # print('统一尺度下',id(self.encode_layers[0]))
        # print('统一尺度下',id(self.encode_layers[1]))
        # print('统一尺度下',id(self.encode_layers[2]))
        #

    
    def forward(self, x,b_rep_node):



        last_g_e=b_rep_node

        B, ts_dim, _,d_model = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)
        #等于说每一个encoder都有deep层，像一个encoder才是下个尺度下的，去看看别的不同尺度是怎样的，哦对，设置成1就ok了
        for attn_layer,dcn_layer in self.encode_layers:

            x = attn_layer(x)   #B D seq_num  d_model
            x = dcn_layer(x, last_g_e)

            # x=self.res_nom(x1+x)





        # x=x.view(B,ts_dim,-1)  #B D (seq_num  d_model)
        # layer_hid=self.linear_hid(x)
        # x=x.view(B,ts_dim,-1,d_model)

        layer_predict = self.linear_pred(x) # B D seq_num  seglen
        layer_predict = torch.transpose(layer_predict,-1,-2)
        layer_predict=self.linear_hid(layer_predict) # B D   seglen  out_num
        layer_predict = torch.transpose(layer_predict, -1, -2) # B D out_num  seglen



        return x,layer_predict   #其中x用于下一层encoder输入 ，后者是当前层的输出









class Encoder(nn.Module):
    '''
    The Encoder of Crossformer.
    '''
    def __init__(self, e_blocks, win_size, d_model, n_heads, d_ff, gcn_dim,gcn_heads,gcn_dropout,block_depth, dropout,out_seg_num,device,\
                 batch,ts_dim,seg_len,in_seg_num = 10,):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList()
        self.Batch=batch
        self.device=device
        self.ts_dim=ts_dim
        self.d_model=d_model
        self.fc = nn.Linear(ts_dim, 16)  ####################奇异值降维

        self.down1 = nn.Linear(d_model, 64)
        self.down2 = nn.Linear(64, 16)

        # self.init_cold=nn.Parameter(torch.zeros(size=[e_blocks,1,ts_dim,gcn_dim],dtype=torch.float))
        self.init_cold=nn.Parameter(torch.zeros(size=[ts_dim,gcn_dim],dtype=torch.float))




        # 这里才是定义多少个尺度的地方
        # 第一层                               Batch, win_size, d_model, n_heads, d_ff, depth, dropout, \
        #                     seg_num = 10
        self.encode_blocks.append(scale_block(self.Batch,1, d_model, n_heads, d_ff, gcn_dim,gcn_heads,gcn_dropout,block_depth,dropout,self.device, \
                                          in_seg_num,out_seg_num=out_seg_num,seg_len=seg_len,ts_dim=ts_dim))
        #后面的大尺度层
        for i in range(1, e_blocks):
            self.encode_blocks.append(scale_block(self.Batch,win_size, d_model, n_heads, d_ff,gcn_dim,gcn_heads,gcn_dropout, block_depth, dropout,self.device, \
                                            ceil(in_seg_num/win_size**i),seg_len=seg_len,ts_dim=ts_dim,out_seg_num=out_seg_num))

        # 不同的尺度下scale块不共享

    def forward(self, x):


        # 承接每一个尺度下的输出
        ts_d=x.shape[1]

        b_rep_node=self.get_init_e(x,device=self.device)


        final_predict = None
        for index,block in enumerate(self.encode_blocks):
            x, layer_predict= block(x,b_rep_node)



            if final_predict is None:
                final_predict=layer_predict
            else:
                final_predict=final_predict+layer_predict

        final_predict = rearrange(final_predict, 'b  out_d seg_num seg_len -> b (seg_num seg_len) out_d', out_d=ts_d)

        return final_predict

    def get_init_e(self, x, device, is_cold_sart=False ):
        '''
        x[b,n,t,d] 通过奇异值分解获得节点表示作为初始的结构信息
        为了节约时间，让所有的encoder_block共用一个
        最后维度也是64
        '''

        b, n, t, d = x.shape

        if is_cold_sart:
            #降低维度减少计算量
            x=self.down1(x)
            x=self.down2(x)



            x = x.view(b, -1, n)
            temp = x.cpu().detach().numpy()


            b_node_rep = [None] * b

            for i in range(b):
                matr = temp[i]
                time_rep, sigam, node_rep = np.linalg.svd(matr)  # node_rep [n,n]
                b_node_rep[i] = torch.tensor(node_rep)



            rep_node = torch.stack(b_node_rep, dim=0).to(device)


            rep_node=self.fc(rep_node)

            # rep_node=self.fc_test(x)


        else:
             # rep_node=self.init_cold.expand(-1,b,-1,-1)   #[块数，B，N，DCN_dim]为了给不同的尺度都有一个好的初始化
             rep_node = self.init_cold.expand( b, -1, -1)  


        return rep_node   #此时


#两个问题，一个是眼前的bug，另外一个是输出长度不能是按照输入来的

if __name__ == '__main__':
    pass
    # model=Encoder(3,2,256,8,512,1,0.2,4,'cuda:0',64,7,6,28)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(total_params)
