from common.model import load_model
from common.utils import get_device

from skimage.measure import marching_cubes
import numpy as np
import mouette as M
import torch
from torch.utils.data import TensorDataset, DataLoader

import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str)
    parser.add_argument("-o", "--output-name", type=str, default="", help="output name")
    parser.add_argument("-d", '--offset', type=float, nargs='+', default=[0.], help="offset of value to print")
    parser.add_argument("-res", "--resolution", type=int, default=100, help="resolution")
    parser.add_argument("-cpu", action="store_true")
    parser.add_argument("-bs", "--batch-size", type=int, default=5000)
    args = parser.parse_args()

    device = get_device(args.cpu)
    print("DEVICE:", device)

    if len(args.output_name)==0:
        args.output_name = args.model.split("/")[-1].split(".pt")[0]

    # sdf = load_model(args.model, device).vanilla_export().to(device)
    sdf = load_model(args.model, device).to(device)

    domain = M.geometry.BB3D(-0.5, -0.5, -0.5, 1.5, 1.5, 1.5)
    res = args.resolution
    L = [np.linspace(domain.min_coords[i], domain.max_coords[i], res) for i in range(3)]
    pts = np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(3,-1).T
    pts = torch.Tensor(pts).to(device)
    pts = DataLoader(TensorDataset(pts), batch_size=args.batch_size)
    values = torch.Tensor()

    for batch in tqdm(pts, total=len(pts)):
        v_batch = sdf(batch[0]).cpu()
        values = torch.concatenate((values, v_batch))

    values = values.reshape((res,res,res)).detach().cpu().numpy()

    for off in args.offset:
        verts,faces,normals,values = marching_cubes(values, level=off)
        values = values[:, np.newaxis]
        m = M.mesh.RawMeshData()
        m.vertices += list(verts)
        m.faces += list(faces)
        m = M.mesh.SurfaceMesh(m)
        normal_attr = m.vertices.create_attribute("normals", float, 3, dense=True)
        normal_attr._data = normals
        values_attr = m.vertices.create_attribute("values", float, 1, dense=True)
        values_attr._data = values

        M.mesh.save(m,f"{args.output_name}_{off}.obj")
        M.mesh.save(m,f"{args.output_name}_{off}.geogram_ascii")
        