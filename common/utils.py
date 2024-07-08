import torch
from torch.utils.data import DataLoader

import mouette as M
from tqdm import tqdm
import numpy as np

def get_device(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    else:
        return torch.device("cuda")
    
def get_BB(X, dim, pad=0.):
    """Get the axis-aligned bounding box of points in tensor X

    Args:
        X (torch.Tensor): input tensor.
        dim (int): whether points are in 2 or 3 dimensions. Ignore other values.

    Returns:
        BoundingBox
    """
    vmin = torch.min(X, dim=0)[0].cpu()
    vmax = torch.max(X, dim=0)[0].cpu()
    bb = M.geometry.AABB(vmin.numpy(), vmax.numpy())
    bb.pad(pad)
    return bb


def forward_in_batches(model, inputs : np.ndarray, device:str, compute_grad:bool=False, batch_size=5_000):
    inputs = torch.Tensor(inputs).to(device)
    inputs = DataLoader(inputs, batch_size=batch_size)
    outputs = []
    grads = []
    for batch in tqdm(inputs, total=len(inputs)):
        batch.requires_grad = compute_grad
        v_batch = model(batch)
        if compute_grad:
            torch.sum(v_batch).backward()
            grads.append(batch.grad.detach().cpu().numpy())
        outputs.append(v_batch.detach().cpu().numpy())
    
    if compute_grad:
        return np.concatenate(outputs), np.concatenate(grads)
    else:
        return np.concatenate(outputs)