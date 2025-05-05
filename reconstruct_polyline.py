import argparse
import numpy as np
import mouette as M

from common.models import load_model
from common.utils import get_device

from skimage.measure import find_contours
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Polyline Reconstruction",
        description="Marching squares algorithm to run on a neural distance fields for 2D data. See 'reconstruct_surface.py' for the same feature in three dimensions."
    )
    parser.add_argument("model", type=str, help="path to the model '.pt' file")
    parser.add_argument("-o", "--output-name", type=str, default="", help="output name")
    parser.add_argument("-iso", '--isovalues', type=float, nargs='+', default=[0.], help="isosurfaces to consider. If a list is provided, will output a polyline for each value")
    parser.add_argument("-res", "--resolution", type=int, default=800, help="grid resolution for the marching squares")
    parser.add_argument("-cpu", action="store_true", help="force CPU computation")
    parser.add_argument("-bs", "--batch-size", type=int, default=5000, help="batch size")
    parser.add_argument("-r", "--range", action="store_true", help="override the -iso argument and run marching squares for each iso in linspace(-0.1, 0.1, 21)")
    parser.add_argument("--merge", action="store_true", help="whether to output polylines as a single .mesh file instead of one file per isovalue")
    args = parser.parse_args()

    device = get_device(args.cpu)
    print("DEVICE:", device)

    if len(args.output_name)==0:
        args.output_name = args.model.split("/")[-1].split(".pt")[0]
    if args.range: args.isovalues = np.linspace(-0.1,0.1,41)

    sdf = load_model(args.model, device)

    domain = M.geometry.AABB((-1., -1.), (1., 1.))
    domain.pad(0.5)
    resX = args.resolution
    resY = round(resX * domain.span.y/domain.span.x)
    X = np.linspace(domain.mini.x, domain.maxi.x, resX)
    Y = np.linspace(domain.mini.y, domain.maxi.y, resY)
    pts = np.hstack((np.meshgrid(X,Y))).swapaxes(0,1).reshape(2,-1).T
    pts = torch.Tensor(pts).to(device)
    pts = DataLoader(TensorDataset(pts), batch_size=args.batch_size)
    dist_values = []

    for (batch,) in tqdm(pts, total=len(pts)):
        batch.requires_grad = False
        v_batch = sdf(batch).cpu()
        dist_values.append(v_batch.detach().cpu().numpy())

    img = np.concatenate(dist_values).reshape((resX,resY))
    print("Value range:", np.min(img), np.max(img))

    if args.merge:
        PL = M.mesh.PolyLine()
        iso_attr = PL.vertices.create_attribute("iso", float)
        id_cnt = 0
        for iso in args.isovalues:
            contours = find_contours(img, level=iso)
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
                    iso_attr[id_cnt + id_v+i] = iso
                id_v += n_cnt
            id_cnt += id_v
        M.mesh.save(PL, f"{args.output_name}.mesh")
        M.mesh.save(PL, f"{args.output_name}.geogram_ascii") # to export the 'iso' attribute
    else:
        for ioff, iso in enumerate(args.isovalues):
            try:
                contours = find_contours(img, level=iso)
                id_v = 0
                for cnt in contours:
                    # Reproject points in real space instead of pixel space
                    n_cnt = len(cnt)
                    PL = M.mesh.PolyLine()
                    for i,(x,y) in enumerate(cnt):
                        px, py = int(x), int(y)
                        dx, dy = x%1, y%1
                        vx = (1-dx)*X[px] + dx * X[px+1]
                        vy = (1-dy)*Y[py] + dy * Y[py+1]
                        PL.vertices.append([vx, vy, 0.])
                        PL.edges.append(sorted((id_v+i, id_v+(i+1)%n_cnt)))
                    id_v += n_cnt
                    M.mesh.save(PL, f"output/{ioff:02d}_{args.output_name}_{round(100*iso)}.mesh")
            except ValueError as e:
                print(e)
                continue
        