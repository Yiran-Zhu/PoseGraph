from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from models.basic_conv import BasicGraphConv
from models.graph_atrous_conv import GraphConv
from models.enc_dec import Enc_Dec

from models.graph.h36m_graph import AdjMatrixGraph
from models.graph.h36m_graph_j import AdjMatrixGraph_J
from models.graph.h36m_graph_b import AdjMatrixGraph_B
from models.graph.h36m_graph_p import AdjMatrixGraph_P


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    if fc.bias:
        nn.init.constant_(fc.bias, 0)


class PoseGTAC(nn.Module):
    """
    PoseGTAC: Graph Transformer Encoder-Decoder with Atrous Convolution.
    """
    def __init__(self, hid_dim, coords_dim=(2, 3), p_dropout=None):
        super(PoseGTAC, self).__init__()

        graph = AdjMatrixGraph()
        adj = torch.from_numpy(graph.A)
        graph_j = AdjMatrixGraph_J()
        graph_p = AdjMatrixGraph_P()
        graph_b = AdjMatrixGraph_B()
        A_binary = graph_j.A_binary
        B_binary = graph_p.A_binary
        C_binary = graph_b.A_binary
        
        self.gconv_input = GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout)
        self.gconv_layers = Enc_Dec(hid_dim, A_binary, B_binary, C_binary, p_dropout=p_dropout)
        self.gconv_output = BasicGraphConv(hid_dim, coords_dim[1], adj)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm1d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)


    def forward(self, x):
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        out = self.gconv_output(out)
        return out
