import torch
from torch import nn
import torch.nn.functional as F

# On the Effectiveness of Weight-Encoded Neural Implicit 3D Shapes
def MultiLayerPerceptron(dim_in, dim_hidden, n_layers, activ=nn.ReLU, final_activ=nn.Identity):
    layers = []
    layers.append(nn.Linear(dim_in, dim_hidden))
    layers.append(activ())

    for _ in range(n_layers-1):
        layers.append(nn.Linear(dim_hidden, dim_hidden))
        layers.append(activ())
    
    layers.append(nn.Linear(dim_hidden, 1))
    layers.append(final_activ())

    model = nn.Sequential(*layers)
    model.meta = [dim_in, dim_hidden, n_layers, activ, final_activ]
    model.id = "MLP"
    return model
    

class MultiLayerPerceptronSkips(nn.Module):

    def __init__(self, dim_in, dim_hidden, n_layers, skips=[]):
        super().__init__()
        self.skips = [x in skips for x in range(n_layers)]
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(dim_in, dim_hidden))
        for i in range(1,n_layers-1):
            if self.skips[i+1]:
                self.layers.append(nn.Linear(dim_hidden, dim_hidden-dim_in))
            else:
                self.layers.append(nn.Linear(dim_hidden,dim_hidden))
        self.last_layer = nn.Linear(dim_hidden,1)
        self.id = "MLPS"
        self.meta = [dim_in, dim_hidden, n_layers, skips]

    def forward(self,x):
        x0 = x
        for k,layer in enumerate(self.layers):
            if self.skips[k]:
                x = layer(torch.concat((x,x0), dim=-1))
            else:
                x = layer(x)
            x = F.relu(x)
        x = self.last_layer(x)
        return x
