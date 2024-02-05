import mouette as M

import os
import numpy as np
import argparse
from igl import signed_distance
from common.visualize import point_cloud_from_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_mesh", type=str, \
        help="path to the input mesh")
    parser.add_argument("--unsigned", action="store_true")
    parser.add_argument("-is", "--importance-sampling", action="store_true")
    parser.add_argument("-n", "--n-train", type=int, default=100_000)
    parser.add_argument("-nt", "--n-test",  type=int, default=10_000)
    parser.add_argument("-visu", help="generates visualization point cloud", action="store_true")
    args = parser.parse_args()

    os.makedirs("inputs", exist_ok=True)

    print("Load mesh")
    mesh = M.mesh.load(args.input_mesh)
    mesh = M.transform.fit_into_unit_cube(mesh)
    mesh = M.transform.translate(mesh, -np.mean(mesh.vertices, axis=0))
    domain = M.geometry.BB3D.of_mesh(mesh,padding=1.)

    print("Generate train set")
    if args.importance_sampling:
        BETA = 50
        print(" | Sampling points")
        X_train_uniform = M.sampling.sample_bounding_box_3D(domain, 10*args.n_train)
        print(" | Compute sampling weights")
        Y_train = signed_distance(X_train_uniform, np.array(mesh.vertices), np.array(mesh.faces, dtype=np.int32))[0]
        weight = np.exp(-BETA*abs(Y_train))
        weight /= np.sum(weight)
        print(" | Importance sampling")
        sampled = np.random.choice(X_train_uniform.shape[0],size=args.n_train,replace=False,p=weight)
        X_train = X_train_uniform[sampled]
        Y_train = Y_train[sampled]
    else:
        n_train_surf = args.n_train//10 # sample 10% of dataset on the surface
        n_train_other = args.n_train - n_train_surf
        print(" | Sampling points on surface")
        X_train_surf = M.sampling.sample_points_from_surface(mesh, n_train_surf)
        print(" | Sampling uniform distribution in domain")
        X_train_other = M.sampling.sample_bounding_box_3D(domain, n_train_other)
        print(" | Compute distances")
        Y_train = signed_distance(X_train_other, np.array(mesh.vertices), np.array(mesh.faces, dtype=np.int32))[0]

        X_train = np.concatenate((X_train_surf, X_train_other))
        Y_train = np.concatenate((np.zeros(n_train_surf), Y_train))

    print("\nGenerate test set")
    n_test_surf = args.n_test // 10
    n_test_other = args.n_test - n_test_surf
    print(" | Sampling points on surface")
    X_surf_test = M.processing.sampling.sample_points_from_surface(mesh, n_test_surf)
    print(" | Sampling uniform distribution in domain")
    X_other_test = M.processing.sampling.sample_bounding_box_3D(domain, n_test_other)
    print(" | Compute distances")
    Y_test = signed_distance(X_other_test, np.array(mesh.vertices), np.array(mesh.faces, dtype=np.int32))[0]
    
    X_test = np.concatenate((X_surf_test, X_other_test))
    Y_test = np.concatenate((np.zeros(n_test_surf), Y_test))
    if args.unsigned:
        Y_train = abs(Y_train)
        Y_test = abs(Y_test)

    if args.visu:
        print("\nGenerate visualization output")
        pc_train = point_cloud_from_array(X_train)
        pc_test = point_cloud_from_array(X_test)
        dist_attr_train = pc_train.vertices.create_attribute("d", float, dense=True)
        dist_attr_test = pc_test.vertices.create_attribute("d", float, dense=True)
        for i in pc_train.id_vertices:
            dist_attr_train[i] = Y_train[i]
        for i in pc_test.id_vertices:
            dist_attr_test[i] = Y_test[i]

    print("Saving files")
    name = M.utils.get_filename(args.input_mesh)
    if args.visu:
        M.mesh.save(pc_train, f"inputs/{name}_pts_train.geogram_ascii")
        M.mesh.save(pc_test, f"inputs/{name}_pts_test.geogram_ascii")
    np.save(f"inputs/{name}_Xtrain.npy", X_train)
    np.save(f"inputs/{name}_Ytrain.npy", Y_train)
    np.save(f"inputs/{name}_Xtest.npy", X_test)
    np.save(f"inputs/{name}_Ytest.npy", Y_test)