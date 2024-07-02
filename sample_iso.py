from common.models import load_model
from common.utils import get_device
from common.visualize import point_cloud_from_array
from common.sdf import *

import argparse
import mouette as M

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Isosurface Sampler',
        description='Sample points that are on a given isosurface of a neural implicit representation')

    parser.add_argument("model", type=str, help="path to the model '.pt' file")
    parser.add_argument("-o", "--output-name", type=str, default="", help="output name")
    parser.add_argument("-iso", '--iso', type=float, default=0., help="value of the isosurface to project on")
    parser.add_argument("-n", "--n-points", type=int, default=1000, help="number of sampled points")
    parser.add_argument("-cpu", action="store_true", help="force CPU computation")
    parser.add_argument("-bs", "--batch-size", type=int, default=1000, help="batch size")
    args = parser.parse_args()

    device = get_device(args.cpu)
    print("DEVICE:", device)

    if len(args.output_name)==0:
        args.output_name = args.model.split("/")[-1].split(".pt")[0]

    model = load_model(args.model, device).to(device)
    DIM = model.meta[0] # input dimension
    if DIM==2:
        domain = M.geometry.BB2D(-1, -1, 1, 1)
        init_samples = M.sampling.sample_bounding_box_2D(domain, args.n_points)
    elif DIM==3:
        init_samples = M.sampling.sample_sphere(M.Vec.zeros(3), 2., args.n_points)
    
    surf_samples = sample_iso(init_samples, model, iso=args.iso, batch_size=args.batch_size, max_steps=10)
    M.mesh.save(point_cloud_from_array(surf_samples),"output/surface_samples.obj")