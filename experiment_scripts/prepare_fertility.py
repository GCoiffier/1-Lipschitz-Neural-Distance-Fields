import mouette as M
import numpy as np
from scipy.spatial import KDTree

fertility = M.mesh.load("/home/guillaume/Desktop/neuralSDF_inputs/input_mesh_signed/fertility.obj")

fertility = M.transform.fit_into_unit_cube(fertility)
fertility = M.transform.translate_to_origin(fertility)

pts_good = M.sampling.sample_points_from_surface(fertility, 5000, return_point_cloud=True,return_normals=True)
M.mesh.save(pts_good, "inputs/fertility_good.geogram_ascii")

pts_sparse = M.sampling.sample_points_from_surface(fertility, 500, return_point_cloud=True, return_normals=True)
M.mesh.save(pts_sparse, "inputs/fertility_sparse.geogram_ascii")

pts_noisy = M.sampling.sample_points_from_surface(fertility, 5000, return_point_cloud=True, return_normals=True)
for v in pts_noisy.id_vertices:
    pts_noisy.vertices[v] = M.Vec(pts_noisy.vertices[v]) + np.random.normal(0., 3e-2, size=3)
M.mesh.save(pts_noisy, "inputs/fertility_noisy.geogram_ascii")

pts_ablated, normals_ablated = M.sampling.sample_points_from_surface(fertility, 5000, return_normals=True)
to_keep = np.ones(5000, dtype=bool)
tree = KDTree(pts_ablated)
for k in range(30):
    while True:
        random_v = np.random.randint(0, 5000)
        if to_keep[random_v] : break
    dist_nn, ind_nn = tree.query(pts_ablated[random_v], 41)
    to_keep[ind_nn] = False

pts_ablated_visu = M.mesh.from_arrays(pts_ablated)
pts_ablated_visu.vertices.register_array_as_attribute("normals", normals_ablated)
pts_ablated_visu.vertices.register_array_as_attribute("keep", to_keep)
M.mesh.save(pts_ablated_visu, "inputs/fertility_ablated_visu.geogram_ascii")

pts_ablated = pts_ablated[to_keep]
normals_ablated = normals_ablated[to_keep]
pts_ablated = M.mesh.from_arrays(pts_ablated)
pts_ablated.vertices.register_array_as_attribute("normals", normals_ablated)
M.mesh.save(pts_ablated, "inputs/fertility_ablated.geogram_ascii")
