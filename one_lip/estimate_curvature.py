import os
import argparse

import mouette as M

from common.dataset import PointCloudDataset
from common.model import *
from common.visualize import *
from common.training import Trainer
from common.utils import get_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="Signed Distance Function")
    parser.add_argument("points", type=str, help="query points on the surface of the model")
    parser.add_argument("-o", "--output-name", type=str, default="")
    parser.add_argument('-bs',"--batch-size", type=int, default=1000, help="Batch size")
    args = parser.parse_args()

    output_folder = os.path.join("output", args.output_name if len(args.output_name)>0 else args.model)
    os.makedirs(output_folder, exist_ok=True)
    device = get_device()
    print("DEVICE:", device)

    #### Load points ####
    points = np.load(args.points)
    print("Loaded point array of shape", points.shape)
    sdf = load_model(args.model)
    print(f"Loaded SDF model ({count_parameters(sdf)} parameters)")

    data = DataLoader(torch.Tensor(points), batch_size=args.batch_size, shuffle=False)
    distances = []
    gradients = []
    for (batch,) in tqdm(data, total=len(data)):
        batch.requires_grad = True
        y = torch.sum(sdf(batch))
        y.backward()
        distances.append(y.detach().cpu().numpy())
        gradients.append(batch.grad.detach().cpu().numpy())
    
    distances = np.concatenate(distances)
    gradients = np.concatenate(gradients, axis=0)

    print(np.min(distances), np.max(distances))
    print(distances.shape, gradients.shape)
