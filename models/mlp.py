import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    This MLP is used to reduce channels and aggregate features in GAC.
    """
    def __init__(self, in_channels, out_channels, dropout=0):
        super(MLP, self).__init__()
        channels = [in_channels] + out_channels
        self.layers = nn.ModuleList()
        for i in range(1, len(channels)):
            if dropout > 0.001:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Conv1d(channels[i-1], channels[i], kernel_size=1))
            self.layers.append(nn.BatchNorm1d(channels[i]))
            self.layers.append(nn.ReLU())

    def forward(self, x):
        # Input shape: (N,V,C)
        x = x.permute(0,2,1)
        for layer in self.layers:
            x = layer(x)
        return x.permute(0,2,1)

