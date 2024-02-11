import numpy as np
import mouette as M
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def point_cloud_from_array(X, D=None):
    pc = M.mesh.PointCloud()
    if X.shape[1]==2:
        X = np.pad(X, ((0,0),(0,1)))
    pc.vertices += list(X)
    if D is not None:
        attr = pc.vertices.create_attribute("dist", float, dense=True)
        attr._data = D.reshape((D.shape[0], 1))
    return pc


def vector_field_from_array(pos, vec, scale=1.) -> M.mesh.PolyLine:
    pl = M.mesh.RawMeshData()
    if pos.shape[1]==2:
        pos = np.pad(pos, ((0,0),(0,1)))
        vec = np.pad(vec, ((0,0),(0,1)))
    n_pt = pos.shape[0]
    for i in range(n_pt):
        P1 = pos[i]
        P2 = pos[i] + scale * vec[i]
        pl.vertices += [P1, P2]
        pl.edges.append((2*i,2*i+1))
    return M.mesh.PolyLine(pl)


def point_cloud_from_arrays(*args) -> M.mesh.PointCloud:
    clouds, labels = [], []
    for pts,label in args:
        clouds.append(point_cloud_from_array(pts))
        labels.append(np.full(pts.shape[0], fill_value=label))
    pc = M.mesh.merge(clouds)
    label_attr = pc.vertices.create_attribute("label", float, dense=True)
    label_attr._data = np.concatenate(labels)[:, np.newaxis]
    return pc


def render_sdf_2d(render_path, contour_path, gradient_path, model, domain : M.geometry.BB2D, device, res=800, batch_size=1000):
    
    with_grad = (gradient_path is not None)

    X = np.linspace(domain.left, domain.right, res)
    resY = round(res * domain.height/domain.width)
    Y = np.linspace(domain.bottom, domain.top, resY)

    pts = np.hstack((np.meshgrid(X,Y))).swapaxes(0,1).reshape(2,-1).T
    pts = torch.Tensor(pts).to(device)
    pts = DataLoader(TensorDataset(pts), batch_size=batch_size)


    dist_values = []
    grad_norms = []

    print()
    for (batch,) in tqdm(pts, total=len(pts)):
        batch.requires_grad = with_grad
        y = model(batch).cpu()
        dist_values.append(y.detach().cpu().numpy())
        if with_grad:
            y = torch.sum(model(batch))
            y.backward()
            grad_norm = torch.norm(batch.grad, dim=1)
            grad_norms.append(grad_norm.detach().cpu().numpy())
        
    img = np.concatenate(dist_values).reshape((res,resY)).T
    img = img[::-1,:]

    vmin = np.amin(img)
    vmax = np.amax(img)
    if vmin>0 or vmax<0:
        vmin,vmax = -1, 1

    if render_path is not None:
        norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        plt.clf()
        pos = plt.imshow(img, cmap="seismic", norm=norm)
        plt.axis('off')
        plt.colorbar(pos)
        plt.savefig(render_path, bbox_inches='tight', pad_inches=0)

    if contour_path is not None:
        plt.clf()
        cs = plt.contourf(X,-Y,img, levels=[-0.01, 0., 0.01], extend="both")
        # cs.cmap.set_over('red')
        # cs.cmap.set_under('blue')
        cs.changed()
        plt.savefig(contour_path, bbox_inches='tight', pad_inches=0)

    if gradient_path is not None:
        grad_img = np.concatenate(grad_norms).reshape((res,resY)).T
        grad_img = grad_img[::-1,:]
        print("GRAD NORM INTERVAL", (np.min(grad_img), np.max(grad_img)))

        plt.clf()
        pos = plt.imshow(grad_img, vmin=0, vmax=2, cmap="seismic")
        plt.axis("off")
        plt.colorbar(pos)
        plt.savefig(gradient_path, bbox_inches='tight', pad_inches=0)


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