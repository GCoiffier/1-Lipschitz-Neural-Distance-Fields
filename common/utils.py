import torch
import mouette as M

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
    if dim == 2:
        bb = M.geometry.BB2D(float(vmin[0]), float(vmin[1]), float(vmax[0]), float(vmax[1]))
        bb.pad(pad,pad)
    elif dim == 3:
        bb = M.geometry.BB3D(*vmin, *vmax)
        bb.pad(pad,pad,pad)
    return bb