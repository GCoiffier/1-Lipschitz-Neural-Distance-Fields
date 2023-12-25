import numpy as np
import os
import mouette as M
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def point_cloud_from_tensor(Xin, Xout) -> M.mesh.PointCloud:
    pc = M.mesh.new_point_cloud()
    pc.vertices += list(Xin)
    pc.vertices += list(Xout)
    attr = pc.vertices.create_attribute("in", bool)
    for i in range(Xin.shape[0]):
        attr[i] = True
    return pc

def render_sdf(path, model, Z, domain, device, res=400, pad=0.2):
    X = np.linspace(domain.min_coords[0]-pad, domain.max_coords[0]+pad, res)
    resY = round(res * (domain.max_coords[1] - domain.min_coords[1])/(domain.max_coords[0]-domain.min_coords[0]))
    Y = np.linspace(domain.min_coords[1]-pad, domain.max_coords[1]+pad, resY)

    img = np.zeros((res,resY))
    for i in range(res):
        inp = torch.Tensor([[X[i], Y[j], Z] for j in range(resY)]).to(device)
        img[i,:] = np.squeeze(model(inp).cpu().detach().numpy())
    img = img[:,::-1].T

    vmin = np.amin(img)
    vmax = np.amax(img)
    if vmin>0 or vmax<0:
        vmin,vmax = -1, 1

    norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
    plt.imshow(img, cmap="seismic", norm=norm)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)