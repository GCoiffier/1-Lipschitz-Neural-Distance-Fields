import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial import KDTree

from .utils import forward_in_batches

def project_onto_iso(
    query_pts : np.ndarray, 
    model : torch.nn.Module, 
    iso : float = 0., 
    device : str = "cpu", 
    batch_size : int = 1000, 
    normalize_grad : bool = False,
    max_steps : int = 100, 
    precision : float = 1e-5
):
    query_tnsr = torch.Tensor(query_pts).to(device)
    loader = DataLoader(query_tnsr, batch_size=batch_size)
    output = []
    for batch in loader:
        for step in range(max_steps):
            batch.requires_grad = True
            dist = model(batch) - iso
            torch.sum(dist).backward()
            grad = batch.grad
            if torch.all( torch.abs(dist) < precision): break
            if normalize_grad:
                batch = batch - (grad / torch.norm(grad, dim=1)**2) * dist
            else:
                batch = batch - grad * dist
            batch = batch.detach()
        output.append(batch.detach().cpu().numpy())
    return np.concatenate(output)


def gradient_descent(
    query_pts : np.ndarray, 
    model : torch.nn.Module, 
    device : str = "cpu", 
    step_size : float = 0.1, 
    batch_size : int = 1000, 
    normalize_grad : bool = False,
    max_steps : int = 100
):
    query_tnsr = torch.Tensor(query_pts).to(device)
    loader = DataLoader(query_tnsr, batch_size=batch_size)
    output = []
    for batch in loader:
        for step in range(max_steps):
            batch.requires_grad = True
            dist = model(batch)
            torch.sum(dist).backward()
            grad = batch.grad
            if normalize_grad:
                batch = batch - step_size*(grad / torch.norm(grad, dim=1)**2)
            else:
                batch = batch - step_size*grad
            batch = batch.detach()
        output.append(batch.detach().cpu().numpy())
    return np.concatenate(output)


def sample_iso(
    init_pts : np.ndarray, 
    model : torch.nn.Module, 
    iso : float = 0.,
    device : str = "cpu",
    k : int = 10,
    stdv : float = 0.2,
    step_size : float = 0.1, 
    batch_size : int = 1000, 
    normalize_grad : bool=False, 
    max_steps : int = 10
):    
    samples = project_onto_iso(init_pts, model, iso=iso, device=device, batch_size=batch_size, normalize_grad=normalize_grad, max_steps=1)  
    for step in range(max_steps):
        print(step)

        ### Compute KDtree and make point repulse themselves
        kdtree = KDTree(samples)
        dist_nn, ind_nn = kdtree.query(samples, k=k+1)
        dist_nn, ind_nn = dist_nn[:,1:], ind_nn[:,1:] # discard self
        k_nn = samples[ind_nn]
        directions = k_nn - samples[:,None,:]

        # project into local tangent plane
        # normals = forward_in_batches(model,samples,device,compute_grad=True, batch_size=batch_size)[1]
        # normals /= np.linalg.norm(normals,axis=1)[:,np.newaxis]
        # directions = directions - np.dot(directions,normals) * normals

        # move the samples
        directions /= np.linalg.norm(directions,axis=2)[:,:,np.newaxis]
        weights = np.exp(-dist_nn**2/stdv)[:,:,None]
        samples = samples - step_size * np.sum(weights * directions,axis=1)

        ### Project back onto surface
        samples = project_onto_iso(samples, model, iso=iso, device=device, batch_size=batch_size, normalize_grad=normalize_grad)
    return samples