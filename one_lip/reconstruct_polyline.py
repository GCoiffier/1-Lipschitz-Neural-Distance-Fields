from common.model import load_model
from common.utils import get_device
from common.visualize import render_sdf

from skimage.measure import find_contours
import numpy as np
import mouette as M
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str)
    parser.add_argument("-o", "--output-name", type=str, default="", help="output name")
    parser.add_argument("-d", '--offset', type=float, nargs='+', default=[0.], help="offset of value to print")
    parser.add_argument("-res", "--resolution", type=int, default=800, help="resolution")
    parser.add_argument("-cpu", action="store_true")
    parser.add_argument("-bs", "--batch-size", type=int, default=5000)
    args = parser.parse_args()

    device = get_device(args.cpu)
    print("DEVICE:", device)

    if len(args.output_name)==0:
        args.output_name = args.model.split("/")[-1].split(".pt")[0]

    # sdf = load_model(args.model, device).vanilla_export().to(device)
    sdf = load_model(args.model, device).to(device)

    domain = M.geometry.BB2D(-0.5,-0.5,1.5,1.5)
    resX = args.resolution
    resY = round(resX * domain.height/domain.width)
    X = np.linspace(domain.left, domain.right, resX)
    Y = np.linspace(domain.bottom, domain.top, resY)
    pts = np.hstack((np.meshgrid(X,Y))).swapaxes(0,1).reshape(2,-1).T
    pts = torch.Tensor(pts).to(device)
    pts = DataLoader(TensorDataset(pts), batch_size=args.batch_size)
    dist_values = []

    for (batch,) in tqdm(pts, total=len(pts)):
        batch.requires_grad = False
        v_batch = sdf(batch).cpu()
        dist_values.append(v_batch.detach().cpu().numpy())

    img = np.concatenate(dist_values).reshape((resX,resY))
    print(np.min(img), np.max(img))

    for off in args.offset:
        contours = find_contours(img, level=off)
        polyline = M.mesh.new_polyline()
        iv = 0
        for cnt in contours:
            # Reproject points in real space instead of pixel space
            n_cnt = len(cnt)
            for i,(x,y) in enumerate(cnt):
                px, py = int(x), int(y)
                dx, dy = x%1, y%1
                vx = (1-dx)*X[px] + dx * X[px+1]
                vy = (1-dy)*Y[py] + dy * Y[py+1]
                polyline.vertices.append([vx, vy, 0.])
                polyline.edges.append(sorted((iv+i, iv+(i+1)%n_cnt)))
            iv += n_cnt
        M.mesh.save(polyline, f"{args.output_name}_{round(100*off)}.mesh")
        