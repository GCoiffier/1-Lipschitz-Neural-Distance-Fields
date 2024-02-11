import torch
from torch import nn

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
    model.meta = [dim_in, dim_hidden, n_layers]
    model.id = "MLP"
    return model
    