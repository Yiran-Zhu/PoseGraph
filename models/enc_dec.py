import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.graph_atrous_conv import GraphAtrousConv as GAC
from models.graph_transformer import GraphTransformerLayer as GTL
from models.scaled_ops import *


class Enc_Dec(nn.Module):
    def __init__(self, dim, A_binary, B_binary, C_binary, p_dropout=None):
        super(Enc_Dec, self).__init__()
        # GAC.
        self.GCN_J = GAC(A_binary, dim, dim, num_scales=4, p_dropout=p_dropout)
        self.Up_GCN_J = GAC(A_binary, 2*dim, dim, num_scales=4, p_dropout=p_dropout)
        self.GCN_P = GAC(B_binary, dim, dim, num_scales=4, p_dropout=p_dropout)
        self.Up_GCN_P = GAC(B_binary, 2*dim, dim, num_scales=4, p_dropout=p_dropout)
        self.GCN_B = GAC(C_binary, dim, dim, num_scales=3, p_dropout=p_dropout)
        # GTL.
        self.tr1 = GTL(dim, dim, dim//4, num_node=16, p_dropout=0)
        self.tr2 = GTL(dim, dim, dim//4, num_node=10, p_dropout=0)
        self.tr3 = GTL(dim, dim, dim//4, num_node=5, p_dropout=0)
        self.tr4 = GTL(dim, dim, dim//4, num_node=10, p_dropout=0)
        self.tr5 = GTL(dim, dim, dim//4, num_node=16, p_dropout=0)
        # Down and Up.
        self.Down_J2P = Down_Joint2Part()
        self.Down_P2B = Down_Part2Body()
        self.Up_B2P = Up_Body2Part()
        self.Up_P2J = Up_Part2Joint()
    
    def forward(self, J_in):
        J_pre = self.GCN_J(J_in)
        J_pre = self.tr1(J_pre)
        down1 = self.Down_J2P(J_pre)
        P_pre = self.GCN_P(down1)
        P_pre = self.tr2(P_pre)
        down2 = self.Down_P2B(P_pre)
        B_pre = self.GCN_B(down2)
        B_pre = self.tr3(B_pre)
        up1 = self.Up_B2P(B_pre)
        P_next = self.Up_GCN_P(torch.cat((P_pre, up1), dim=-1))
        P_next = self.tr4(P_next)
        up2 = self.Up_P2J(P_next)
        J_next = self.Up_GCN_J(torch.cat((J_pre, up2), dim=-1))
        J_next = self.tr5(J_next)

        return J_next