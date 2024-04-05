from common.models import load_model
from common.utils import get_device
from common.visualize import point_cloud_from_array
from common.sdf import *

import argparse
import mouette as M

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str)
    parser.add_argument("-o", "--output-name", type=str, default="", help="output name")
    parser.add_argument("-iso", '--isovalues', type=float, default=0.)
    parser.add_argument("-n", "--n-points", type=int, default=1000, help="number of sampled points")
    parser.add_argument("-cpu", action="store_true")
    parser.add_argument("-bs", "--batch-size", type=int, default=1000)
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
        # domain = M.geometry.BB3D(-1, -1, -1, 1, 1, 1)
        # init_samples = M.sampling.sample_bounding_box_3D(domain, args.n_points)
        init_samples = M.sampling.sample_sphere(M.Vec.zeros(3), 2., args.n_points)
    
    # surf_samples = project_onto_iso(init_samples, model, iso=0., batch_size=args.batch_size, max_steps=100, device=device)
    # surf_samples = gradient_descent(init_samples, model, batch_size=args.batch_size, max_steps=100)
    surf_samples = sample_iso(init_samples, model, batch_size=args.batch_size, max_steps=10)
    M.mesh.save(point_cloud_from_array(surf_samples),"output/surface_samples.obj")