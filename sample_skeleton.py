from common.models import load_model
from common.utils import get_device
from common.visualize import point_cloud_from_array
from common.sdf import *

import argparse
import mouette as M

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Skeleton Sampler',
        description='Sample points that are near the skeleton of a 1-Lipschitz implicit neural network')

    parser.add_argument("model", type=str, help="path to the model '.pt' file")
    parser.add_argument("-o", "--output-name", type=str, default="", help="output name")
    parser.add_argument("-res", '--res', type=int, default=300, help="resolution of the grid to sample")
    parser.add_argument("-cpu", action="store_true", help="force cpu compute")
    parser.add_argument("-bs", "--batch-size", type=int, default=10_000, help="batch size")
    args = parser.parse_args()

    device = get_device(args.cpu)
    print("DEVICE:", device)

    if len(args.output_name)==0:
        args.output_name = args.model.split("/")[-1].split(".pt")[0]

    model = load_model(args.model, device).to(device) 
    skeleton_samples = sample_skeleton(model, args.res, device=device, batch_size=args.batch_size, descent_steps=10)
    M.mesh.save(point_cloud_from_array(skeleton_samples),"output/skeleton_samples.obj")