import mouette as M
from mouette import geometry as geom

import os
import numpy as np
from numpy.random import choice
import argparse
from scipy.spatial import KDTree
from common.visualize import point_cloud_from_array, point_cloud_from_arrays

def sample_points(mesh, n_pts):
    sampled_pts = np.zeros((n_pts, 3))
    lengths = M.attributes.edge_length(mesh, persistent=False).as_array()
    lengths /= np.sum(lengths)
    if len(mesh.edges)==1:
        edges = [0]*n_pts
    else:
        edges = choice(len(mesh.edges), size=n_pts, p=lengths)
    for i,e in enumerate(edges):
        pA,pB = (mesh.vertices[_v] for _v in mesh.edges[e])
        t = np.random.random()
        pt = t*pA + (1-t)*pB
        sampled_pts[i,:] = pt 
    return sampled_pts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Dataset Generator",
        description="Generate a 3D dataset from a 3D polyline. Only outputs a dataset for an unsigned distance field."
    )

    parser.add_argument("input_mesh", type=str, \
        help="path to the input mesh")
    parser.add_argument("-no", "--n-train", type=int, default=10000, help="number of samples in the training dataset")
    parser.add_argument("-nt", "--n-test",  type=int, default=3000, help="number of samples in the testing dataset.")
    parser.add_argument("-visu", help="generates visualization point cloud", action="store_true")
    args = parser.parse_args()

    os.makedirs("inputs", exist_ok=True)

    print("Load mesh")
    input_mesh = M.mesh.load(args.input_mesh)
    input_mesh = M.transform.fit_into_unit_cube(input_mesh)
    input_mesh = M.transform.translate_to_origin(input_mesh)

    if isinstance(input_mesh, M.mesh.SurfaceMesh):
        print("Extract boundary polyline")
        mesh, _ = M.processing.extract_curve_boundary(input_mesh)
    elif isinstance(input_mesh, M.mesh.PolyLine):
        mesh = input_mesh

    domain = M.geometry.BB3D.of_mesh(mesh)
    arrays_to_save = dict()
    mesh_to_save = dict() # if args.visu

    print("Generate train set")
    print(" | Sampling points")
    X_on  = sample_points(mesh, args.n_train)
    X_out1 = M.sampling.sample_bounding_box_3D(domain, 4*args.n_train//5)
    domain.pad(0.5, 0.5, 0.5)
    X_out2 = M.sampling.sample_bounding_box_3D(domain, args.n_train - 4*args.n_train//5)
    X_out = np.concatenate((X_out1, X_out2))
    np.random.shuffle(X_out)
    print(f" | Generated {X_on.shape[0]} (on), {X_out.shape[0]} (outside)")
    arrays_to_save["Xtrain_on"] = X_on
    arrays_to_save["Xtrain_out"] = X_out
    if args.visu:
        mesh_to_save["pts_on"] = point_cloud_from_array(X_on)
        mesh_to_save["pts_train"] = point_cloud_from_arrays((X_on, -1), (X_out, 1))

    print("Generate test set")
    X_test = M.sampling.sample_bounding_box_3D(domain, args.n_test)
    tree = KDTree(np.concatenate((mesh.vertices, X_on)))
    Y_test,_ = tree.query(X_test, workers=-1)
    arrays_to_save["Xtest"] = X_test
    arrays_to_save["Ytest"] = Y_test
    if args.visu:
        mesh_to_save["test"] = point_cloud_from_array(X_test, Y_test)

    print("Saving files")
    name = M.utils.get_filename(args.input_mesh)
    if args.visu:
        for mesh_name, mesh in mesh_to_save.items():
            M.mesh.save(mesh, f"inputs/{name}_{mesh_name}.geogram_ascii")
    for ar_name,ar in arrays_to_save.items():
        np.save(f"inputs/{name}_{ar_name}.npy", ar)