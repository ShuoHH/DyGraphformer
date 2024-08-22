import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from collections import OrderedDict

class fc_layer(nn.Module):
    def __init__(self, in_channels, out_channels, need_layer_norm):
        super(fc_layer, self).__init__()
        self.linear_w = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_uniform_(self.linear_w.data)

        self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=[1, 1], bias=True)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.need_layer_norm = need_layer_norm

    def forward(self, input):
        '''
        input = batch_size, in_channels, nodes, time_step
        output = batch_size, out_channels, nodes, time_step
        '''
        if self.need_layer_norm:
            result = F.leaky_relu(torch.einsum('bani,io->bano ', [input.transpose(1, -1), self.linear_w]))\
                     # + self.layer_norm(self.linear(input).transpose(1, -1))
        else:
            result = F.leaky_relu(torch.einsum('bani,io->bano ', [input.transpose(1, -1), self.linear_w])) \
                     # + self.linear(input).transpose(1, -1)
        return result.transpose(1, -1)

class DGCN(nn.Module):
    '''

    DGCN是抽取了t时刻的时空嵌入和当前时刻X


    '''
    def __init__(self,batch, embed_dim,gcn_dim,ts_dim,gcn_heads):
        super(DGCN, self).__init__()
        self.heads=gcn_heads
        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.embed_dim = embed_dim
        self.weights_z = nn.Parameter(torch.randn(size=[2*gcn_dim,gcn_dim],dtype=torch.float))
        self.weights_r = nn.Parameter(torch.randn(size=[2*gcn_dim,gcn_dim],dtype=torch.float))
        self.weights_h = nn.Parameter(torch.randn(size=[2*gcn_dim,gcn_dim],dtype=torch.float))
        # self.conv=dgl.nn.pytorch.GraphConv(16,16)
        # self.graph=dgl.DGLGraph()
        self.gcn=GCN(embed_dim,embed_dim)

        self.x_nom=nn.LayerNorm(embed_dim)
        self.skip_norm = nn.LayerNorm(ts_dim)
        self.D = self.heads * gcn_dim  # node_dim #

        self.query = fc_layer(in_channels=gcn_dim, out_channels=self.D, need_layer_norm=False)
        self.key = fc_layer(in_channels=gcn_dim, out_channels=self.D, need_layer_norm=False)
        self.value = fc_layer(in_channels=gcn_dim, out_channels=self.D, need_layer_norm=False)

        self.mlp = nn.Conv2d(in_channels=self.heads, out_channels=self.heads, kernel_size=(1, 1), bias=True)
        self.attn_norm = nn.LayerNorm(ts_dim)
        self.last_fc= nn.Linear(ts_dim,gcn_dim)
        self.linear_norm=nn.LayerNorm(ts_dim)

        self.attn_linear = nn.Parameter(torch.zeros(size=(ts_dim, ts_dim)))
        nn.init.xavier_uniform_(self.attn_linear.data, gain=1.414)
        self.attn_linear_1 = nn.Parameter(torch.zeros(size=(ts_dim, ts_dim)))
        nn.init.xavier_uniform_(self.attn_linear_1.data, gain=1.414)
        self.attn_norm_1=nn.LayerNorm(ts_dim)
        self.bn=nn.LayerNorm(gcn_dim)


        self.last_nom=nn.LayerNorm(embed_dim)
        self.supports1 = torch.eye(ts_dim).to('cuda:0')


        self.fc=nn.Sequential( #疑问，这里为什么要用三层linear来做，为什么激活函数是sigmoid
                OrderedDict([('fc1', nn.Linear(gcn_dim, self.hyperGNN_dim)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid1', nn.Sigmoid()),
                             ('fc2', nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid2', nn.Sigmoid()),
                             ('fc3', nn.Linear(self.middle_dim, gcn_dim))]))
    def forward(self, orig_x,x, last_G_emb,graph_s=None):
        '''
        node_embed是那个结合时间的时空嵌入
        '''
        #x shaped[B, N, C], last_G_e shaped [B,N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        batch,node_num,gcn_dim =x.shape
        #添加图的节点数
        #创建单位矩阵

        x = self.fc(x)
        ######获得当前时刻图向量 x||last_g_e   [64,7,128]   [b,n,2e][b,2e,e]
        tmp = torch.cat((x,last_G_emb ), dim=-1)
        upgate_z=torch.sigmoid(torch.einsum('bnm,me->bne',tmp,self.weights_z))
        upgate_r = torch.sigmoid(torch.einsum('bnm,me->bne',tmp,self.weights_r))

        #图嵌入向量的当前时刻输入
        temp=torch.mul(upgate_r,last_G_emb)
        temp2=torch.einsum('bnm,me->bne', torch.cat([temp, x], dim=-1), self.weights_h)

        #这个生成的有问题全是0
        H_g_t=torch.tanh(temp2)

        H_g_e=torch.mul((1-upgate_z),last_G_emb)+torch.mul(upgate_z,H_g_t)



        H_g_e=self.bn(H_g_e)########################
        last = H_g_e

        soc=torch.matmul(x, x.transpose(2, 1))
        skip_attn=self.skip_norm(soc)
        H_g_e = H_g_e.unsqueeze(1).transpose(1, -1)#################################

        query=self.query(H_g_e)
        key=self.key(H_g_e)
        query = query.squeeze(-1).contiguous().transpose(1,2).view(batch,-1, self.heads, gcn_dim)  #b,n,h,g

        key = key.squeeze(-1).contiguous().transpose(1,2).view(batch,-1, self.heads, gcn_dim)  #b,n,h,g
        attention = torch.einsum('bhnd, bhdu-> bhnu', query.transpose(1,2), key.transpose(1,2).transpose(-1,-2))

        attention /= (gcn_dim ** 0.5)


        attention = self.mlp(attention) + attention
        attention=torch.sum(attention, dim=1)

        adj_bf = self.attn_norm(attention) + skip_attn

        orig_x=self.x_nom(orig_x)########################

        supports2 = DGCN.get_laplacian(F.relu(adj_bf), self.supports1)
        x_g2=self.gcn(orig_x,supports2)
        x_g1 = torch.einsum("nm,bmc->bnc", self.supports1, orig_x)

        x_g2=self.last_nom(x_g1+x_g2)

        # x_g2 = torch.stack([x_g1, x_g2], dim=1)



        return x_g2,last

    @staticmethod
    def get_laplacian(graph, I, normalize=True):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            #L = I - torch.matmul(torch.matmul(D, graph), D)
            L = torch.matmul(torch.matmul(D, graph), D)
        else:
            graph = graph + I
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.matmul(torch.matmul(D, graph), D)
        return L


class GraphConvolution(nn.Module):
    '''
    Simple GCN layer
    '''

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features


        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)  # xavier初始化，

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj):
        # inputs: (N, n_channels), adj: sparse_matrix (N, N)
        input = torch.matmul(input, self.weight)
        output = torch.einsum("bnn,bnc->bnc", adj,input )  # 原来版本的图卷积
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super(GCN, self).__init__()

        self.first_layer = GraphConvolution(n_features, hidden_dim)
        self.hid_layer = GraphConvolution(n_features, hidden_dim)
        # self.hid_layer2 = GraphConvolution(n_features, hidden_dim)不好

        self.last_layer = GraphConvolution(n_features,hidden_dim)
        # self.dropout=nn.Dropout(0.2)不好


    def forward(self, inputs, adj):
        x = self.first_layer(inputs, adj)

        x = self.hid_layer(x, adj)





        x = self.last_layer(x, adj)
        return x

if __name__ == '__main__':
    pass

