from common.models import load_model
from common.utils import get_device
from common.visualize import reconstruct_surface_marching_cubes

import mouette as M
from sphere_tracing import sdf
import torch
from scipy.spatial.transform import Rotation

device = get_device()
print("DEVICE:", device)

bunny = load_model("/home/coiffier/Bureau/SDF_results/0067_stanford_bunny/model_final.pt", device)

SPHERE_CENTER3 = torch.Tensor([0., 0.5, -0.1]).to(device)
SPHERE_RADIUS3 = 0.4
sphere3 = sdf.translation(sdf.sphere(SPHERE_RADIUS3), SPHERE_CENTER3)

octopus = load_model("/home/coiffier/Bureau/SDF_results/0066_octopus/model_e100.pt", device)
OCTOPUS_POS = torch.Tensor([0., -0.3, 0.2]).to(device)
OCTOPUS_EULER = [0, 0, 0]
OCTOPUS_SCALE = 2.
rot = torch.Tensor(Rotation.from_euler("XYZ", OCTOPUS_EULER, degrees=True).as_matrix()).to(device)

octopus = sdf.rounding(octopus, -5e-3)
octopus = sdf.intersection(octopus, sdf.complementary(sphere3))
octopus = sdf.translation(sdf.rotation( sdf.scaling(octopus, OCTOPUS_SCALE), rot), OCTOPUS_POS)
F = sdf.smooth_union(bunny, octopus, k=0.1)

domain = M.geometry.BB3D(-2, -2, -2, 2, 2, 2)
for surface in reconstruct_surface_marching_cubes(F, domain, device, iso=0, res=400).values():
    M.mesh.save(surface,f"output/csg.obj")
    break

