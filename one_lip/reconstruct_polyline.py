from common.model import load_model
from common.utils import get_device
from common.visualize import render_sdf

from skimage.measure import find_contours
import numpy as np
import mouette as M
import torch
import argparse
from tqdm import trange

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str)
    parser.add_argument("-o", "--output-name", type=str, default="", help="output name")
    parser.add_argument("-d", '--offset', type=float, nargs='+', default=[0.], help="offset of value to print")
    parser.add_argument("-res", "--resolution", type=int, default=800, help="resolution")
    parser.add_argument("-cpu", action="store_true")
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

    img = np.zeros((resX,resY))

    for i in trange(resX):
        inp = torch.Tensor([[X[i], Y[j]] for j in range(resY)]).to(device)
        img[i,:] = np.squeeze(sdf(inp).cpu().detach().numpy())

    print(np.min(img), np.max(img))

    for off in args.offset:
        contours = find_contours(img, level=off)
        polyline = M.mesh.new_polyline()
        for cnt in contours:
            # Reproject points in real space instead of pixel space
            for x,y in cnt:
                px, py = int(x), int(y)
                dx, dy = x%1, y%1
                vx = (1-dx)*X[px] + dx * X[px+1]
                vy = (1-dy)*Y[py] + dy * Y[py+1]
                polyline.vertices.append([vx, vy, 0.])
            n = len(polyline.vertices)
            for i in range(n):
                polyline.edges.append(sorted((i,(i+1)%n)))
        M.mesh.save(polyline, f"{args.output_name}_{off}.mesh")
        