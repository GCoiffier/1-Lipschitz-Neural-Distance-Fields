import numpy as np
import os
import mouette as M
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def point_cloud_from_tensor(Xin, Xout) -> M.mesh.PointCloud:
    pc = M.mesh.new_point_cloud()
    pc.vertices += list(np.pad(Xin,((0,0),(0,1)))) # don't forget to add z=0 to coordinates
    pc.vertices += list(np.pad(Xout,((0,0),(0,1))))
    attr = pc.vertices.create_attribute("in", bool)
    for i in range(Xin.shape[0]):
        attr[i] = True
    return pc

def render_sdf(path, model, xrange, yrange, device, res=800, pad=0.2):
    X = np.linspace(xrange[0]-pad, xrange[1]+pad, res)
    resY = round(res * (yrange[1] - yrange[0])/(xrange[1]-xrange[0]))
    Y = np.linspace(yrange[0]-pad, yrange[1]+pad, resY)

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
    plt.imshow(img, cmap="seismic", norm=norm)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)