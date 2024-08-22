import torch
import torch.nn as nn
from dygraph_models.DDGCN import DGCN
import math

class DGCRM(nn.Module):
    def __init__(self,ts_dim,batch, d_model,gcn_dim,gcn_heads,dropout=0.2):
        super(DGCRM, self).__init__()

        self.DGCR=DGCN(ts_dim=ts_dim,batch=batch,embed_dim=d_model,gcn_dim=gcn_dim,gcn_heads=gcn_heads)

        self.down_dim=nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Linear(64, gcn_dim)
        )

        self.up_dim = nn.Sequential(
            nn.Linear(gcn_dim, 64),
            nn.Linear(64, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)  #留着MLP的时候用
        self.dropout=nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # for _ in range(1, num_layers):
        #     self.DGCRM_cells.append(DDGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))
        self.MLP1 = nn.Sequential(nn.Linear(d_model, 512),
                                nn.GELU(),
                                nn.Linear(512, d_model))


    def forward(self, x, last_g_e):
        #shape of x: (B, N, T, D)
        #shape of last_g_e (b,n,d_model)

        seq_length = x.shape[2]  #[64,7,28,256]

        current_inputs = self.down_dim(x) #[64,7,28,64]
        # current_inputs = x
        all_graph=[]
        output_hidden=[]

        for t in range(seq_length):   #如果有两层GRU，则第二层的GGRU的输入是前一层的隐藏状态
            x_g2,H_g_e = self.DGCR(x[:, :, t, :],current_inputs[:, :, t, :],last_g_e,)  #current_inputs[:, :, t, :] (B,N,D) last_g_e :(B,N,D)
            last_g_e=H_g_e


            output_hidden.append(x_g2)

        output_hidden=torch.stack(output_hidden,dim=2)


        # #注意力操作 点乘所有图 B D segnum d_model
        output_hidden=output_hidden+self.dropout(output_hidden)
        output_hidden = self.norm1(output_hidden)
        output_hidden=output_hidden+self.dropout(self.MLP1(output_hidden))
        output_hidden=self.norm2(output_hidden)






        return  output_hidden



if __name__ == '__main__':
    pass
    # model=DGCRM(64,256)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(total_params)



