import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.basic_conv import BasicGraphConv
from models.mlp import MLP
from models.graph.tools import k_adjacency, normalize_adjacency_matrix


class GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(GraphConv, self).__init__()

        self.gconv = BasicGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x)
        if isinstance(x, tuple):
            x = x[0]
        x = x.transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class GraphAtrousConv(nn.Module):
    """
    Graph Atrous Convolution with Skip.
    """
    def __init__(self, A_binary, input_dim, output_dim, num_scales, p_dropout):
        super(GraphAtrousConv, self).__init__()
        self.num_scales = num_scales

        A_powers = [k_adjacency(A_binary, k, with_self=True) for k in range(num_scales)]
        A_powers = [torch.from_numpy(normalize_adjacency_matrix(g)) for g in A_powers]
        self.gconv = nn.ModuleList()
        for i in range(0, self.num_scales):
            self.gconv.append(nn.Sequential(GraphConv(A_powers[i], input_dim, output_dim, p_dropout)))

        self.mlp = MLP((num_scales+1)*output_dim, [output_dim])

        if input_dim != output_dim:
            self.down = nn.Sequential(
                nn.Conv1d(input_dim, output_dim, 1),
                nn.BatchNorm1d(output_dim),
                # relu or not relu.
                nn.ReLU()
            )
        else:
            self.down = lambda x: x

    def forward(self, x):
        residual = self.down(x.permute(0,2,1)).permute(0,2,1)

        out = []
        for i in range(self.num_scales):
            out.append(self.gconv[i](x))
        out = torch.cat(out, dim=2)
        out = self.mlp(out)

        return residual + out