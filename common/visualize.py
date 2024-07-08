import numpy as np
import mouette as M
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from skimage.measure import marching_cubes

from .utils import forward_in_batches

def point_cloud_from_array(X, D=None):
    pc = M.mesh.from_arrays(X)
    if D is not None:
        pc.vertices.register_array_as_attribute("dist",D)
    return pc

def point_cloud_from_arrays(*args) -> M.mesh.PointCloud:
    clouds, labels = [], []
    for pts,label in args:
        clouds.append(point_cloud_from_array(pts))
        labels.append(np.full(pts.shape[0], fill_value=label))
    pc = M.mesh.merge(clouds)
    pc.vertices.register_array_as_attribute("label", np.concatenate(labels))
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


def render_sdf_2d(render_path, contour_path, gradient_path, model, domain : M.geometry.AABB, device, res=1000, batch_size=1000):
    """ Renders a 2D SDF

    Args:
        render_path (str): Path to export a color render
        contour_path (str): Path to export the contour plot
        gradient_path (str): Path to export the gradient norm plot
        model : SDF neural model
        domain (M.geometry.AABB): axis aligned bounding box of dimension 2to represent the plotting domain.
        device (str): cpu or cuda
        res (int, optional): Image resolution. Defaults to 800.
        batch_size (int, optional): Size of forward batches. Defaults to 1000.        
    """
    assert domain.dim == 2

    X = np.linspace(domain.mini[0], domain.maxi[0], res)
    resY = round(res * domain.span[1]/domain.span[0])
    Y = np.linspace(domain.mini[1], domain.maxi[1], resY)

    pts = np.hstack((np.meshgrid(X,Y))).swapaxes(0,1).reshape(2,-1).T
    if gradient_path is not None:
        dist_values,grad_values = forward_in_batches(
            model, pts, device, 
            compute_grad=True, batch_size=batch_size)
    else:
        dist_values = forward_in_batches(model, pts, device, compute_grad=False, batch_size=batch_size)

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
        norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        plt.imshow(img, cmap="bwr", norm=norm)
        plt.axis("off")
        # cs = plt.contourf(X,-Y,img, levels=np.linspace(-0.1,0.1,11), cmap="seismic", extend="both")
        # cs.changed()
        plt.contour(img, levels=16, colors='k', linestyles="solid", linewidths=0.3)
        plt.contour(img, levels=[0.], colors='k', linestyles="solid", linewidths=0.6)
        plt.savefig(contour_path, bbox_inches='tight', pad_inches=0, dpi=200)

    if gradient_path is not None:
        grad_norms = np.linalg.norm(grad_values,axis=1)
        grad_img = grad_norms.reshape((res,resY)).T
        grad_img = grad_img[::-1,:]
        print("GRAD NORM INTERVAL", (np.min(grad_img), np.max(grad_img)))

        plt.clf()
        pos = plt.imshow(grad_img, vmin=0.5, vmax=1.5, cmap="bwr")
        plt.contour(img, levels=[0.], colors='k', linestyles="solid", linewidths=0.6)
        plt.axis("off")
        plt.colorbar(pos)
        plt.savefig(gradient_path, bbox_inches='tight', pad_inches=0)

def render_sdf_quad(render_path, contour_path, gradient_path, model, P0, P1, P2, device, res=800, batch_size=1000):
    """
    Renders a 3D sdf along a specified (planar) quad in space.

    Args:
        render_path (str): Path to export a color render
        contour_path (str): Path to export the contour plot
        gradient_path (str): Path to export the gradient norm plot
        model : SDF neural model
        P0 (array): First point of the quad ( (0,0) in parameter space)
        P1 (array): Second point of the quad ( (1,0) in parameter space)
        P2 (array): Thirs point of the quad ( (0,1) in parameter space)
        device (str): cpu or cuda
        res (int, optional): Image resolution. Defaults to 800.
        batch_size (int, optional): Size of forward batches. Defaults to 1000.
    """
    
    dx = P1 - P0
    dy = P2 - P0
    X = np.linspace(0,1, res)
    resY = round(res * M.geometry.norm(dy)/M.geometry.norm(dx))
    Y = np.linspace(0,1, resY)

    pts = []
    for ax in X:
        for ay in Y:
            p = P0 + ax*dx + ay*dy
            pts.append(p)
    pts = np.array(pts)
    
    if gradient_path is not None:
        dist_values,grad_values = forward_in_batches(
            model, pts, device, 
            compute_grad=True, batch_size=batch_size)
    else:
        dist_values = forward_in_batches(model, pts, device, compute_grad=False, batch_size=batch_size)


    img = np.concatenate(dist_values).reshape((res,resY)).T
    img = img[::-1,:]

    vmin = np.amin(img)
    vmax = np.amax(img)
    if vmin>0 or vmax<0:
        vmin,vmax = -1, 1
    print(vmin, vmax)

    if render_path is not None:
        norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        plt.clf()
        pos = plt.imshow(img, cmap="seismic", norm=norm)
        plt.axis('off')
        plt.colorbar(pos)
        plt.savefig(render_path, bbox_inches='tight', pad_inches=0)

    if contour_path is not None:
        plt.clf()
        norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        plt.imshow(img, cmap="bwr", norm=norm)
        plt.axis("off")
        # cs = plt.contourf(X,-Y,img, levels=np.linspace(-0.1,0.1,11), cmap="seismic", extend="both")
        # cs.changed()
        plt.contour(img, levels=16, colors='k', linestyles="solid", linewidths=0.3)
        plt.contour(img, levels=[0.], colors='k', linestyles="solid", linewidths=0.6)
        plt.savefig(contour_path, bbox_inches='tight', pad_inches=0, dpi=200)

    if gradient_path is not None:
        grad_norms = np.linalg.norm(grad_values,axis=1)
        grad_img = grad_norms.reshape((res,resY)).T
        grad_img = grad_img[::-1,:]
        print("GRAD NORM INTERVAL", (np.min(grad_img), np.max(grad_img)))

        plt.clf()
        pos = plt.imshow(grad_img, vmin=0.5, vmax=1.5, cmap="bwr")
        plt.contour(img, levels=[0.], colors='k', linestyles="solid", linewidths=0.6)
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


def reconstruct_surface_marching_cubes(model, domain, device, iso=0, res=100, batch_size=5000):
    if isinstance(iso, (int,float)): iso = [iso]
    
    ### Feed grid to model
    L = [np.linspace(domain.mini[i], domain.maxi[i], res) for i in range(3)]
    pts = np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(3,-1).T
    dist_values = forward_in_batches(model, pts, device, compute_grad=False, batch_size=batch_size)
    dist_values = dist_values.reshape((res,res,res))

    ### Call marching cubes
    to_save = dict()
    for ioff,off in enumerate(iso):
        try:
            verts,faces,normals,values = marching_cubes(dist_values, level=off)
            values = values[:, np.newaxis]
            m = M.mesh.RawMeshData()
            m.vertices += list(verts)
            m.faces += list(faces)
            m = M.mesh.SurfaceMesh(m)
            normal_attr = m.vertices.create_attribute("normals", float, 3, dense=True)
            normal_attr._data = normals
            values_attr = m.vertices.create_attribute("values", float, 1, dense=True)
            values_attr._data = values
            to_save[(ioff, off)] = m
        except ValueError:
            continue
    
    ### Reproject meshes to correct coordinates
    for key,mesh in to_save.items():
        for v in mesh.id_vertices:
            pV = M.Vec(mesh.vertices[v])
            ix, iy, iz = int(pV.x), int(pV.y), int(pV.z)
            dx, dy, dz = pV.x%1, pV.y%1, pV.z%1
            vx = (1-dx)*L[0][ix] + dx * L[0][ix+1]
            vy = (1-dy)*L[1][iy] + dy * L[1][iy+1]
            vz = (1-dz)*L[2][iz] + dz * L[2][iz+1]
            mesh.vertices[v] = M.Vec(vx,vy,vz)
        to_save[key] = mesh
    return to_save