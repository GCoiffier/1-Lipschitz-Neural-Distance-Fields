import torch
from .mlp import MultiLayerPerceptron
from .lipschitz import DenseLipNetwork
from .sll import DenseSDP
from .siren import SirenNet

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, path):
    data = { "id": model.id, "meta" : model.meta, "state_dict" : model.state_dict()}
    torch.save(data, path)
    
def load_model(path, device:str):
    data = torch.load(path, map_location=device)
    model_type = data.get("id","Spectral")
    if model_type == "Spectral":
        model = DenseLipNetwork(*data["meta"])
    elif model_type == "SDP":
        model = DenseSDP(*data["meta"])
    elif model_type == "MLP":
        model = MultiLayerPerceptron(*data["meta"])
    elif model_type == "SIREN":
        model = SirenNet(*data["meta"])
    else:
        raise Exception(f"Model type {model_type} not recognized")
    model.load_state_dict(data["state_dict"])
    return model