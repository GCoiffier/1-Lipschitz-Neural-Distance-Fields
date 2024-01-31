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
    sdf = load_model(args.model, device)
    print(f"Loaded SDF model ({count_parameters(sdf)} parameters)")

    data = DataLoader(torch.Tensor(points), batch_size=args.batch_size, shuffle=False)
    distances = []
    gradients = []
    for batch in tqdm(data, total=len(data)):
        batch.requires_grad = True
        y = sdf(batch)
        ysum = torch.sum(y)
        ysum.backward()
        distances.append(y.detach().cpu().numpy())
        gradients.append(batch.grad.detach().cpu().numpy())
    
    distances = np.squeeze(np.concatenate(distances))
    gradients = np.concatenate(gradients, axis=0)
    print(distances.shape, gradients.shape)

    out_point_cloud = M.mesh.new_point_cloud()
    out_point_cloud.vertices += [pt for pt in points]

    out_grad = M.mesh.new_polyline()
    for iv in range(points.shape[0]):
        v1 = points[iv,:]
        v2 = v1 + 0.01*gradients[iv,:]
        out_grad.vertices += [v1,v2]
        out_grad.edges.append((2*iv,2*iv+1))

    dist_attr = out_point_cloud.vertices.create_attribute("dist", float)
    for v in out_point_cloud.id_vertices:
        dist_attr[v] = distances[v]

    M.mesh.save(out_grad, os.path.join(output_folder, "grad.mesh"))
    M.mesh.save(out_point_cloud, os.path.join(output_folder, "point_cloud.geogram_ascii"))
