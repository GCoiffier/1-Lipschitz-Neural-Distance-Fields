import argparse
import numpy as np
import mouette as M

from common.models import load_model
from common.utils import get_device
from common.visualize import reconstruct_surface_marching_cubes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Surface Reconstruction",
        description="Marching cube algorithm to run on a neural distance fields for 3D surfaces. See 'reconstruct_polyline.py' for the same feature in two dimensions."
    )

    parser.add_argument("model", type=str, help="path to the model '.pt' file")
    parser.add_argument("-o", "--output-name", type=str, default="", help="output name")
    parser.add_argument("-iso", '--isovalues', type=float, nargs='+', default=[0.], help="isosurfaces to consider. If a list is provided, will output a surface mesh for each value")
    parser.add_argument("-res", "--resolution", type=int, default=100, help="grid resolution to consider for marching cubes")
    parser.add_argument("-cpu", action="store_true", help="force CPU computation")
    parser.add_argument("-bs", "--batch-size", type=int, default=5000, help="batch size")
    parser.add_argument("-r", "--range", action="store_true", help="override the -iso argument and run marching cube for each iso in linspace(-0.1, 0.1, 21)")
    args = parser.parse_args()

    device = get_device(args.cpu)
    print("DEVICE:", device)

    if len(args.output_name)==0:
        args.output_name = args.model.split("/")[-1].split(".pt")[0]
    if args.range: 
        args.isovalues = np.linspace(-0.1,0.1,21)

    sdf = load_model(args.model, device)
    domain = M.geometry.AABB((-1, -1, -1), (1, 1, 1))
    res = args.resolution
    meshes = reconstruct_surface_marching_cubes(sdf, domain, device, args.isovalues, args.resolution, args.batch_size)
    for (n,off),mesh in meshes.items():
        M.mesh.save(mesh,f"output/{n:02d}_{args.output_name}_{round(1000*off)}.obj")

