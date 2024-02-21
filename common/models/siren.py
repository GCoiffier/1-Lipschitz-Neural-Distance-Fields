import torch
from torch import nn
import numpy as np

# Code adapted from https://github.com/MClemot/SkeletonLearning/blob/main/siren_nn.py

class SinusActivation(nn.Module):
    def __init__(self, w0 = 30.):
        super().__init__()
        self.w0 = w0 
    def forward(self, x):
        return torch.sin(self.w0*x)   
    
class SirenLayer(nn.Module):
    def __init__(self, 
        dim_in:int, 
        dim_out:int, 
        w0 : float = 1., 
        c : float = 6., 
        is_first_layer:bool = False, 
        activation:bool=True
    ):
        super().__init__()
        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out)
        w_std = (1 / dim_in) if is_first_layer else (np.sqrt(c / dim_in) / w0)
        weight.uniform_(-w_std, w_std)
        bias.uniform_(-w_std, w_std)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.activation = SinusActivation(w0) if activation else nn.Identity()

    def forward(self, x):
        return self.activation(nn.functional.linear(x, self.weight, self.bias))


def SirenNet(dim_in, dim_hidden, n_layers, w0:float = 6., w0_first_layer:float = 30.):
    layers = []
    # First dim_in -> dim_hidden layer
    layers.append(SirenLayer(dim_in, dim_hidden, w0_first_layer, is_first_layer=True))
    
    # Intermediate dim_hidden -> dim_hidden layers
    for _ in range(n_layers-1):
        layers.append(SirenLayer(dim_hidden, dim_hidden, w0))
    
    # last dim_hidden->1 layer has no activation
    layers.append(SirenLayer(dim_in = dim_hidden, dim_out = 1, w0 = w0, activation = False))
    
    model = nn.Sequential(*layers)
    model.id = "SIREN"
    model.meta = [dim_in, dim_hidden, n_layers]
    return model