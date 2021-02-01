import torch
import torch.nn as nn
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Positional Encoding.
    """
    def __init__(self, channel, joint_num):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num

        pos_list = []
        for j_id in range(self.joint_num):
            pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        pe = torch.zeros(self.joint_num, channel)

        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1,0).unsqueeze(0)  # 1 C V
        self.register_buffer('pe', pe)

    def forward(self, x):  # N C V
        x = x + self.pe
        
        return x


class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer Layer.
    """
    def __init__(self, in_channels, out_channels, inter_channels, num_head=3, num_node=16, 
                 stride=1, use_global=True, use_pe=True, p_dropout=0):
        super(GraphTransformerLayer, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_head = num_head
        self.use_global = use_global
        self.use_pes = use_pe

        atts = torch.zeros((1, num_head, num_node, num_node))
        self.register_buffer('atts', atts)
        self.pes = PositionalEncoding(in_channels, num_node)
        self.in_nets = nn.Conv1d(in_channels, 2 * num_head * inter_channels, 1, bias=True)

        if use_global:
            self.global_att = nn.Parameter(torch.ones(1, num_head, num_node, num_node) / num_node)

        self.out_nets = nn.Sequential(
            nn.Conv1d(in_channels * num_head, out_channels, 1, bias=True),
            nn.BatchNorm1d(out_channels),
        )

        if in_channels != out_channels or stride != 1:
            self.downs1 = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm1d(out_channels),
            )
            
        else:
            self.downs1 = lambda x: x
            
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        N, V, C = x.size()
        x = x.permute(0,2,1)

        attention = self.atts
        if self.use_pes:
            y = self.pes(x)
        else:
            y = x
        q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_head, self.inter_channels, V), 2, dim=1)  
        attention = attention + self.soft(torch.einsum('nscu,nscv->nsuv', [q, k]) / (self.inter_channels))
        if self.use_global:
            attention = attention + self.global_att.repeat(N, 1, 1, 1)
        attention = self.drop(attention)
        y = torch.einsum('ncu,nsuv->nscv', [x, attention]).contiguous().view(N, self.num_head * self.in_channels, V)
        y = self.out_nets(y) 

        y = self.relu(self.downs1(x) + y)
        y = y.permute(0,2,1)
        return y


if __name__ == '__main__':
    GTL = GraphTransformer(2, 64, 16, num_head=3, num_node=16, stride=1, \
                use_global=True, use_pe=True, p_dropout=0)
    pose = torch.rand([8,16,2])
    print(GTL(pose).size())
