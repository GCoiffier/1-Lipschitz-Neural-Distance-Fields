from common.models import load_model
from common.utils import get_device
from torch_sphere_tracer import renderers, csg, sdf

import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def main(model, device, resolution, camera_pos, convergence_threshold):
    num_iterations = 1000

    # ---------------- camera matrix ---------------- #

    cx, cy = resolution
    camera_matrix = torch.Tensor([[2*cx, 0.0, cx], [0.0, 2*cy, cy], [0.0, 0.0, 1.0]]).to(device)
    # ---------------- camera position ---------------- #

    distance, azimuth, elevation = camera_pos

    camera_position = torch.Tensor([
        np.sin(elevation) * np.cos(azimuth), 
        np.sin(elevation) * np.sin(azimuth),
        np.cos(elevation)
    ]).to(device) * distance

    # ---------------- camera rotation ---------------- #

    target_position = torch.Tensor([0.0, 0.0, 0.0]).to(device)
    up_direction = torch.Tensor([0.0, 0.0, -1.0]).to(device)

    camera_z_axis = target_position - camera_position
    camera_x_axis = torch.cross(up_direction, camera_z_axis, dim=-1)
    camera_y_axis = torch.cross(camera_z_axis, camera_x_axis, dim=-1)
    camera_rotation = torch.stack((camera_x_axis, camera_y_axis, camera_z_axis), dim=-1)
    camera_rotation = nn.functional.normalize(camera_rotation, dim=-2)

    # ---------------- directional light ---------------- #

    light_directions = torch.Tensor([0.5, 1.0, 0.2]).to(device)

    # ---------------- ray marching ---------------- #
    
    y_positions = torch.arange(cy * 2, dtype=camera_matrix.dtype, device=device)
    x_positions = torch.arange(cx * 2, dtype=camera_matrix.dtype, device=device)
    y_positions, x_positions = torch.meshgrid(y_positions, x_positions)
    z_positions = torch.ones_like(y_positions)
    ray_positions = torch.stack((x_positions, y_positions, z_positions), dim=-1)
    ray_positions = torch.einsum("mn,...n->...m", torch.inverse(camera_matrix),  ray_positions)
    ray_positions = torch.einsum("mn,...n->...m", camera_rotation, ray_positions) + camera_position
    ray_directions = nn.functional.normalize(ray_positions - camera_position, dim=-1)

    # ---------------- rendering ---------------- #

    # ground = sdf.plane(torch.tensor([0.0, -1.0, 0.0], device=device), 0.0)
    # SDF = csg.union(model, ground)
    # SDF = csg.union(sdf.translation(csg.smooth_subtraction(sdf.sphere(1.0), sdf.rounding(sdf.box(0.7), 0.1), 0.05), torch.tensor([0.0, -0.7, 0.0], device=device)), ground)
    SDF = model

    print("Perform Ray Marching")
    surface_positions, converged = renderers.sphere_tracing(
        signed_distance_function=SDF, 
        ray_positions=ray_positions, 
        ray_directions=ray_directions, 
        num_iterations=num_iterations, 
        convergence_threshold=convergence_threshold,
        bounding_radius=1.
    )
    surface_positions = torch.where(converged, surface_positions, torch.zeros_like(surface_positions))

    surface_normals = renderers.compute_normal(
        signed_distance_function=SDF, 
        surface_positions=surface_positions,
    )

    surface_normals = torch.where(converged, surface_normals, torch.zeros_like(surface_normals))

    print("Render Image")
    image = renderers.phong_shading(
        surface_normals=surface_normals, 
        view_directions=camera_position - surface_positions, 
        light_directions=light_directions, 
        light_ambient_color=torch.ones(1, 1, 3, device=device),
        light_diffuse_color=torch.ones(1, 1, 3, device=device), 
        light_specular_color=torch.ones(1, 1, 3, device=device), 
        material_ambient_color=torch.Tensor((0.15, 0.3, 0.8)).reshape((1,1,3)).to(device),
        material_diffuse_color=torch.Tensor((0.5, 0.5, 0.5)).reshape((1,1,3)).to(device),
        material_specular_color=torch.Tensor((0.1, 0.1, 0.1)).reshape((1,1,3)).to(device),
        material_emission_color=torch.zeros(1, 1, 3, device=device),
        material_shininess=64.0,
    )

    # grounded = torch.abs(ground(surface_positions)) < convergence_threshold
    # image = torch.where(grounded, torch.full_like(image, 0.9), image)

    # print("Render Shadows")
    # shadowed = renderers.compute_shadows(
    #     signed_distance_function=SDF, 
    #     surface_positions=surface_positions, 
    #     surface_normals=surface_normals,
    #     light_directions=light_directions, 
    #     num_iterations=num_iterations, 
    #     convergence_threshold=convergence_threshold,
    #     foreground_masks=converged,
    # )
    # image = torch.where(shadowed, image * 0.5, image)
    image = torch.where(converged, image, torch.ones_like(image))

    image = image.detach().cpu().numpy()
    image = image[::-1, ::-1]
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sphere Tracing")

    parser.add_argument("model", type=str)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("-d", "--distance", type=float, default=1.5)
    parser.add_argument("-az", "--azimuth", type=float, default=0.)
    parser.add_argument("-el", "--elevation", type=float, default=0.)
    parser.add_argument("-t", "--convergence-threshold", type=float, default=1e-3)
    parser.add_argument("-o", "--output-name", default="raymarched")
    
    args = parser.parse_args()

    device = get_device()
    model = load_model(args.model, device)

    # image = main(model, device, (args.width, args.height), (args.distance, 4.8, 3.9), args.convergence_threshold) # Elephant parameters
    image = main(sdf.rounding(model, -0.005), device, (args.width, args.height), (args.distance, np.pi/2+0.1, np.pi/2+0.1), args.convergence_threshold)
    plt.imsave(f"{args.output_name}.png", image)
