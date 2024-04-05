"""
Code adapted from the original PHASE implementation

https://github.com/drv-agwl/Implicit_Neural_Representation/tree/master
"""

import numpy as np
import torch.nn as nn
import torch
from math import pi

def doubleWellPotential(s):
    """
    double well potential function with zeros at -1 and 1
    """
    return (s ** 2) - 2 * (s.abs()) + 1.

class FourierLayer(nn.Module):
    def __init__(self, in_features, out_features, k):
        super().__init__()
        B = torch.randn(in_features, out_features) * k
        self.register_buffer("B", B)

    def forward(self, x):
        x_proj = torch.matmul(2 * pi * x, self.B)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out

class PhaseNet(nn.Module):
    def __init__(
            self,
            dim_in, 
            dim_hidden, 
            n_layers,
            FF:bool=True,
            skip_in=(),
            geometric_init=True,
            radius_init=1,
            beta=100
    ):
        super().__init__()
        self.FF = FF
        self.k = 6
        if FF:
            self.ffLayer = FourierLayer(in_features=dim_in, out_features=dim_hidden//2, k=self.k)
            dims = [dim_hidden]*(n_layers-1) + [1]
        else:
            dims = [dim_in] + [dim_hidden]*(n_layers-1) + [1]

        self.n_layers = n_layers
        self.skip_in = skip_in

        for layer in range(0, self.n_layers):
            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - dim_in
            else:
                out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)
            
            if geometric_init: # if true perform geometric initialization
                if layer == self.n_layers - 1:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            setattr(self, "lin" + str(layer), lin)
        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else: # vanilla relu
            self.activation = nn.ReLU()

        self.id = "PHASE"
        self.meta = [dim_in, dim_hidden, n_layers, FF, skip_in]

    def forward(self, input):
        x = input
        if self.FF:
            x = self.ffLayer(x)  # apply the fourier
        for layer in range(0, self.n_layers):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)
            x = lin(x)
            if layer < self.n_layers - 1:
                x = self.activation(x)
        return x
