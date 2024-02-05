import torch
from torch import nn
from deel import torchlip
from .sll_layer import SDPBasedLipschitzDense

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, layers, path):
    data = { "id": model.id, "layers" : layers, "state_dict" : model.state_dict()}
    torch.save(data, path)
    
def load_model(path, device):
    data = torch.load(path, map_location=device)
    model_type = data.get("id","Spectral")
    if model_type == "Spectral":
        model = DenseLipNetwork(data["layers"])
    elif model_type == "SDP":
        model = DenseSDPLip(*data["layers"])
    elif model_type == "MLP":
        model = MultiLayerPerceptron(data["layers"])
    model.load_state_dict(data["state_dict"])
    return model
    
def DenseLipNetwork(
    widths:list, 
    group_sort_size:int=0, 
    k_coeff_lip:float=1., 
    niter_spectral:int=3,
    niter_bjorck:int=15
):
    layers = []
    activation = torchlip.FullSort if group_sort_size == 0 else lambda : torchlip.GroupSort(group_sort_size)
    for ilayer, (w_in, w_out) in enumerate(widths):
        if w_out==1:
            layers.append(torchlip.FrobeniusLinear(w_in, w_out))
        else:
            layers.append(torchlip.SpectralLinear(w_in, w_out, niter_spectral=niter_spectral, niter_bjorck=niter_bjorck))
            layers.append(activation())
    model = torchlip.Sequential(*layers, k_coef_lip=k_coeff_lip)
    model.archi = widths
    model.id = "Spectral"
    return model

def DenseSDPLip(n_in, n_hidden, n_layers):
    layers = []
    layers.append(nn.ZeroPad1d((0, n_hidden-n_in)))
    for ilayer in range(n_layers):
        layers.append(SDPBasedLipschitzDense(n_hidden))
    layers.append(torchlip.FrobeniusLinear(n_hidden,1))
    model = torch.nn.Sequential(*layers)
    model.archi = [n_in, n_hidden, n_layers]
    model.id = "SDP"
    return model

def MultiLayerPerceptron(widths, activ=nn.ReLU):
    """
    On the Effectiveness of Weight-Encoded Neural Implicit 3D Shapes
    """
    layers = []
    for w_in,w_out in widths:
        layers.append(nn.Linear(w_in,w_out))
        if w_out==1:
            layers.append(nn.Tanh())
        else:
            layers.append(activ())
    model = nn.Sequential(*layers)
    model.archi = widths
    model.id = "MLP"
    return model
    