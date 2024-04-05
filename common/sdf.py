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
    precision : float = 1e-6
):
    query_tnsr = torch.Tensor(query_pts).to(device)
    model = model.to(device)
    loader = DataLoader(query_tnsr, batch_size=batch_size)
    output = []
    for batch in loader:
        for _ in range(max_steps):
            batch.requires_grad = True
            dist = model(batch) - iso
            dist.backward(torch.ones_like(dist))
            grad = batch.grad
            if torch.all( torch.abs(dist) < precision): break
            if normalize_grad:
                gnorm = torch.norm(grad, dim=1).unsqueeze(1)
                grad /= gnorm
            batch = batch - grad * dist
            batch = batch.detach()
        output.append(batch.detach().cpu().numpy())
    return np.concatenate(output)


def gradient_descent(
    query_pts : np.ndarray, 
    model : torch.nn.Module, 
    device : str = "cpu", 
    step_size : float = 1.,
    batch_size : int = 1000, 
    use_dist : bool = False,
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
                grad = grad / torch.norm(grad, dim=1)
            if use_dist:
                batch = batch - step_size*dist*grad
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
    samples = project_onto_iso(init_pts, model, iso=iso, device=device, batch_size=batch_size, normalize_grad=normalize_grad, max_steps=100)  
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
        samples = project_onto_iso(samples, model, iso=iso, device=device, batch_size=batch_size, normalize_grad=normalize_grad, max_steps=10)
    return samples



def sample_skeleton(model, res, device : str = "cpu", batch_size=1000, margin=-1e-2, grad_norm_threshold=0.5, descent_steps : int = 0):
    L = [np.linspace(-0.6, 0.6, res) for i in range(3)]
    pts = np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(3,-1).T
    distances = forward_in_batches(model, pts, device, compute_grad=False, batch_size=batch_size)
    distances = np.squeeze(distances)
    pts = pts[distances<margin,:]
    _,grad = forward_in_batches(model, pts, device, compute_grad=True, batch_size=batch_size)
    grad_norm = np.linalg.norm(grad,axis=1)
    pts = pts[grad_norm<grad_norm_threshold]

    if descent_steps>0:
        pts = gradient_descent(pts, model, device, step_size=1e-2, batch_size=batch_size, max_steps=descent_steps)
    return pts