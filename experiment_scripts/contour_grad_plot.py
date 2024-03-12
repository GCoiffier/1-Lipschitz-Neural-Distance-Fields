import mouette as M
from common.visualize import render_sdf_2d
from common.models import load_model
import sys

model = load_model(sys.argv[1], "cpu")

render_sdf_2d(
    None, "contours.png", "gradient.png",
    model, M.geometry.BB2D(-1,-1,1,1), "cpu", 2000, batch_size=1000
)