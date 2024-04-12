from common.models import load_model
from common.utils import get_device
from common.visualize import reconstruct_surface_marching_cubes

import mouette as M
from sphere_tracing import sdf
import torch
from scipy.spatial.transform import Rotation

device = get_device()
print("DEVICE:", device)


SPHERE_RADIUS1 = 0.25
SPHERE_CENTER1 = torch.Tensor([-0.35, 0.3, 0.1]).to(device)
sphere1 = sdf.translation(sdf.sphere(SPHERE_RADIUS1), SPHERE_CENTER1)

SPHERE_RADIUS2 = 0.41 
SPHERE_CENTER2 = torch.Tensor([-0.3, 0.6, -0.2]).to(device)
sphere2 = sdf.translation(sdf.sphere(SPHERE_RADIUS2), SPHERE_CENTER2)

bunny = load_model("/home/coiffier/Bureau/SDF_results/0067_stanford_bunny/model_final.pt", device)
bunny = sdf.smooth_intersection(bunny, sdf.complementary(sphere1), 0.03)
bunny = sdf.intersection(bunny, sdf.complementary(sphere2))

SPHERE_CENTER3 = torch.Tensor([0., 0.5, -0.1]).to(device)
SPHERE_RADIUS3 = 0.4
sphere3 = sdf.translation(sdf.sphere(SPHERE_RADIUS3), SPHERE_CENTER3)

octopus = load_model("/home/coiffier/Bureau/SDF_results/0066_octopus/model_e100.pt", device)
OCTOPUS_POS = torch.Tensor([-0.3, 0.17, 0.05]).to(device)
OCTOPUS_EULER = [-90, 30, -90]
OCTOPUS_SCALE = 1.7
rot = torch.Tensor(Rotation.from_euler("XYZ", OCTOPUS_EULER, degrees=True).as_matrix()).to(device)

octopus = sdf.intersection(octopus, sdf.complementary(sphere3))
octopus = sdf.translation(sdf.rotation( sdf.scaling(octopus, OCTOPUS_SCALE), rot), OCTOPUS_POS)

F = sdf.smooth_union(bunny, octopus, k=0.03)

domain = M.geometry.BB3D(-1, -1, -1, 1, 1, 1)
for surface in reconstruct_surface_marching_cubes(F, domain, device, iso=0, res=300).values():
    M.mesh.save(surface,f"output/csg.obj")
    break

