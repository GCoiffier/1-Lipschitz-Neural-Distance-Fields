import torch
from torch import nn
from deel import torchlip

def DenseLipNetwork(widths, group_sort_size:int=0, k_coeff_lip:float=1.0):
    layers = []
    activation = torchlip.FullSort() if group_sort_size == 0 else torchlip.GroupSort(group_sort_size)
    for w_in, w_out in widths:
        layers.append(torchlip.SpectralLinear(w_in, w_out))
        if w_out!=1:
            layers.append(activation)
    model = torchlip.Sequential(*layers, k_coef_lip=k_coeff_lip)
    return model

def MultiLayerPerceptron(widths, activ=nn.ReLU):
    layers = []
    for w_in,w_out in widths:
        layers.append(nn.Linear(w_in,w_out))
        layers.append(activ())
    model = nn.Sequential(*layers)
    return model
    