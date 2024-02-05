import numpy as np
import mouette as M
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def point_cloud_from_array(X):
    pc = M.mesh.PointCloud()
    if X.shape[1]==2:
        X = np.pad(X, ((0,0),(0,1)))
    pc.vertices += list(X)
    return pc

def point_cloud_from_tensors(Xin, Xout, Xbd=None) -> M.mesh.PointCloud:
    pc = M.mesh.new_point_cloud()
    pc_in = point_cloud_from_array(Xin)
    pc_out = point_cloud_from_array(Xout)
    if Xbd is not None:
        pc_bd = point_cloud_from_array(Xbd)
        pc = M.mesh.merge([pc_in, pc_out, pc_bd])
    else:
        pc = M.mesh.merge([pc_in, pc_out])
    attr = pc.vertices.create_attribute("in", int, default_value=-1, dense=True)
    n = Xin.shape[0]
    for i in range(Xout.shape[0]):
        attr[n+i] = 1
    if Xbd is not None:
        n = Xin.shape[0] + Xout.shape[0]
        for i in range(Xbd.shape[0]):
            attr[n+1] = 0
    return pc

def render_sdf(path, model, domain : M.geometry.BB2D, device, res=800, batch_size=1000):
    X = np.linspace(domain.left, domain.right, res)
    resY = round(res * domain.height/domain.width)
    Y = np.linspace(domain.bottom, domain.top, resY)

    pts = np.hstack((np.meshgrid(X,Y))).swapaxes(0,1).reshape(2,-1).T
    pts = torch.Tensor(pts).to(device)
    pts = DataLoader(TensorDataset(pts), batch_size=batch_size)
    dist_values = []

    dist_values = []
    print()
    for (batch,) in tqdm(pts, total=len(pts)):
        batch.requires_grad = False
        v_batch = model(batch).cpu()
        dist_values.append(v_batch.detach().cpu().numpy())
        
    img = np.concatenate(dist_values).reshape((res,resY)).T
    img = img[::-1,:]

    vmin = np.amin(img)
    vmax = np.amax(img)
    if vmin>0 or vmax<0:
        vmin,vmax = -1, 1

    norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
    plt.clf()
    pos = plt.imshow(img, cmap="seismic", norm=norm)
    plt.axis('off')
    plt.colorbar(pos)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)

def render_gradient_norm(path, model, domain : M.geometry.BB2D, device, res=800, batch_size=1000):
    X = np.linspace(domain.left, domain.right, res)
    resY = round(res * domain.height/domain.width)
    Y = np.linspace(domain.bottom, domain.top, resY)

    pts = np.hstack((np.meshgrid(X,Y))).swapaxes(0,1).reshape(2,-1).T
    pts = torch.Tensor(pts).to(device)
    pts = DataLoader(TensorDataset(pts), batch_size=batch_size)
    grad_norms = []
    print()
    for (batch,) in tqdm(pts, total=len(pts)):
        batch.requires_grad = True
        y = torch.sum(model(batch))
        y.backward()
        grad_norm = torch.norm(batch.grad, dim=1)
        grad_norms.append(grad_norm.detach().cpu().numpy())
    img = np.concatenate(grad_norms).reshape((res,resY)).T
    img = img[::-1,:]
    print("GRAD NORM INTERVAL", (np.min(img), np.max(img)))

    plt.clf()
    pos = plt.imshow(img, vmin=0, vmax=2, cmap="seismic")
    plt.axis("off")
    plt.colorbar(pos)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)

def parameter_singular_values(model):
    layers = list(model.children())
    data= []
    for layer in layers:
        if hasattr(layer, "weight"):
            w = layer.weight
            u, s, v = torch.linalg.svd(w)
            # data.append(f"{layer}, {s}")
            data.append(f"{layer}, min={s.min()}, max={s.max()}")
    return data