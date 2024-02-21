from common.models import load_model
from common.utils import get_device
from common.visualize import reconstruct_surface_marching_cubes

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
    parser.add_argument("-iso", '--isovalues', type=float, nargs='+', default=[0.], help="offset of value to print")
    parser.add_argument("-res", "--resolution", type=int, default=100, help="resolution")
    parser.add_argument("-cpu", action="store_true")
    parser.add_argument("-bs", "--batch-size", type=int, default=5000)
    parser.add_argument("-r", "--range", action="store_true", help="output isolines for linspace(-0.1, 0.1, 21)")
    args = parser.parse_args()

    device = get_device(args.cpu)
    print("DEVICE:", device)

    if len(args.output_name)==0:
        args.output_name = args.model.split("/")[-1].split(".pt")[0]
    if args.range: 
        args.isovalues = np.linspace(-0.05,0.05,21)

    # sdf = load_model(args.model, device).vanilla_export().to(device)
    sdf = load_model(args.model, device).to(device)
    domain = M.geometry.BB3D(-1, -1, -1, 1, 1, 1)
    res = args.resolution
    meshes = reconstruct_surface_marching_cubes(sdf, domain, device, args.isovalues, args.resolution, args.batch_size)
    for (n,off),mesh in meshes.items():
        M.mesh.save(mesh,f"output/{n:02d}_{args.output_name}_{round(1000*off)}.obj")

