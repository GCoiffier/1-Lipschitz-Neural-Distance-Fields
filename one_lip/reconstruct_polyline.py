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
    parser.add_argument("-r", "--range", action="store_true", help="output isolines for linspace(-0.1, 0.1, 31)")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    device = get_device(args.cpu)
    print("DEVICE:", device)

    if len(args.output_name)==0:
        args.output_name = args.model.split("/")[-1].split(".pt")[0]
    if args.range: args.offset = np.linspace(-0.1,0.1,31)

    # sdf = load_model(args.model, device).vanilla_export().to(device)
    sdf = load_model(args.model, device).to(device)

    domain = M.geometry.BB2D(-1, -1, 1, 1)
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

    if args.merge:
        PL = M.mesh.new_polyline()
        iso = PL.vertices.create_attribute("iso", float)
        id_cnt = 0
        for off in args.offset:
            contours = find_contours(img, level=off)
            id_v = 0
            for cnt in contours:
                # Reproject points in real space instead of pixel space
                n_cnt = len(cnt)
                for i,(x,y) in enumerate(cnt):
                    px, py = int(x), int(y)
                    dx, dy = x%1, y%1
                    vx = (1-dx)*X[px] + dx * X[px+1]
                    vy = (1-dy)*Y[py] + dy * Y[py+1]
                    PL.vertices.append([vx, vy, 0.])
                    PL.edges.append(sorted((id_cnt+id_v+i, id_cnt+id_v+(i+1)%n_cnt)))
                    iso[id_cnt + id_v+i] = off
                id_v += n_cnt
            id_cnt += id_v
        M.mesh.save(PL, f"{args.output_name}.mesh")
        M.mesh.save(PL, f"{args.output_name}.geogram_ascii")
    else:
        for ioff, off in enumerate(args.offset):
            try:
                contours = find_contours(img, level=off)
                id_v = 0
                for cnt in contours:
                    # Reproject points in real space instead of pixel space
                    n_cnt = len(cnt)
                    PL = M.mesh.new_polyline()
                    for i,(x,y) in enumerate(cnt):
                        px, py = int(x), int(y)
                        dx, dy = x%1, y%1
                        vx = (1-dx)*X[px] + dx * X[px+1]
                        vy = (1-dy)*Y[py] + dy * Y[py+1]
                        PL.vertices.append([vx, vy, 0.])
                        PL.edges.append(sorted((id_v+i, id_v+(i+1)%n_cnt)))
                    id_v += n_cnt
                    M.mesh.save(PL, f"output/{ioff:02d}_{args.output_name}_{round(100*off)}.mesh")
            except ValueError:
                continue
        