import numpy as np
import mouette as M
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def point_cloud_from_array(X):
    pc = M.mesh.PointCloud()
    if X.shape[1]==2:
        X = np.pad(X, ((0,0),(0,1)))
    pc.vertices += list(X)
    return pc

def point_cloud_from_tensor(Xin, Xout) -> M.mesh.PointCloud:
    pc = M.mesh.new_point_cloud()
    if Xin.shape[1]==2:
        # add z=0 to coordinates
        pc.vertices += list(np.pad(Xin,((0,0),(0,1)))) 
        pc.vertices += list(np.pad(Xout,((0,0),(0,1))))
    else:
        pc.vertices += list(Xin)
        pc.vertices += list(Xout)
    attr = pc.vertices.create_attribute("in", bool)
    for i in range(Xin.shape[0]):
        attr[i] = True
    return pc

def render_sdf(path, model, domain : M.geometry.BB2D, device, res=800, z=0.):
    X = np.linspace(domain.left, domain.right, res)
    resY = round(res * domain.height/domain.width)
    Y = np.linspace(domain.bottom, domain.top, resY)

    img = np.zeros((res,resY))

    for i in range(res):
        inp = torch.Tensor([[X[i], Y[j]] for j in range(resY)]).to(device)
        img[i,:] = np.squeeze(model(inp).cpu().detach().numpy())
    img = img[:,::-1].T

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

def render_gradient_norm(path, model, domain : M.geometry.BB2D, device, res=800):
    X = np.linspace(domain.left, domain.right, res)
    resY = round(res * domain.height/domain.width)
    Y = np.linspace(domain.bottom, domain.top, resY)
    img = np.zeros((res,resY))

    gmin, gmax = float("inf"), -float("inf")
    for i in range(res):
        inp = torch.Tensor([[X[i], Y[j]] for j in range(resY)]).to(device)
        inp.requires_grad = True
        y = model(inp)
        Gy = torch.ones_like(y)
        y.backward(Gy) #,retain_graph=True)
        # retrieve gradient of the function
        grad = inp.grad
        grad_norm = torch.sqrt(torch.sum(grad**2, axis=1))
        print(grad.shape, grad_norm.shape)
        exit()
        gmin = min(gmin, float(torch.min(grad_norm)))
        gmax = max(gmax, float(torch.max(grad_norm)))
        img[i,:] = np.squeeze(grad_norm.cpu().detach().numpy())
    img = img[:,::-1].T

    print("GRAD NORM INTERVAL", (gmin, gmax))

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
            u, s, v = torch.svd(w)
            data.append(f"{layer}, min={s.min()}, max={s.max()}")
    return data