import mouette as M
from common.visualize import render_sdf_2d, render_sdf_quad
from common.models import load_model
import sys
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = load_model(sys.argv[1], device)

if len(sys.argv)>2:
    quad = M.mesh.load(sys.argv[2])
    P0 = quad.vertices[0]
    P1 = quad.vertices[1]
    P2 = quad.vertices[2]
    render_sdf_quad(
        None, "contours.png", "gradient.png",
        model, P0, P1, P2, device, 
        res=2000, batch_size=10_000
    )

else:
    render_sdf_2d(
        None, "contours.png", "gradient.png",
        model, M.geometry.BB2D(-1,-1,1,1), device, 
        res=2000, batch_size=10_000
    )