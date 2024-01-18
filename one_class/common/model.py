import torch
from torch import nn
from deel import torchlip

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, layers, path):
    data = { "layers" : layers, "state_dict" : model.state_dict()}
    torch.save(data, path)
    
def load_model(path):
    data = torch.load(path)
    model = DenseLipNetwork(data["layers"])
    model.load_state_dict(data["state_dict"])
    return model
    
def DenseLipNetwork(
    widths:list, 
    group_sort_size:int=0, 
    k_coeff_lip:float=1.0, 
    niter_spectral:int=3, 
    niter_bjorck:int=15
):
    layers = []
    activation = torchlip.FullSort() if group_sort_size == 0 else torchlip.GroupSort(group_sort_size)
    for w_in, w_out in widths:
        if w_out==1:
            layers.append(torchlip.FrobeniusLinear(w_in, w_out))
        else:
            layers.append(torchlip.SpectralLinear(w_in, w_out, niter_spectral=niter_spectral, niter_bjorck=niter_bjorck))
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
    