import mouette as M

import os
import numpy as np
import argparse
from igl import fast_winding_number_for_meshes, signed_distance
from common.visualize import point_cloud_from_array, point_cloud_from_arrays, vector_field_from_array

def extract_train_point_cloud(n_surf, n_train, mesh, domain):
    X_bd, N_bd = M.processing.sampling.sample_points_from_surface(mesh, n_surf, return_normals=True)
    domain.pad(0.05,0.05,0.05)
    X_other1 = M.processing.sampling.sample_bounding_box_3D(domain, 50*n_train)
    domain.pad(0.95,0.95,0.95)
    X_other2 = M.processing.sampling.sample_bounding_box_3D(domain, 5*n_train)
    X_other = np.concatenate((X_other1, X_other2))
    np.random.shuffle(X_other)
    Y_other = fast_winding_number_for_meshes(np.array(mesh.vertices), np.array(mesh.faces, dtype=np.int32), X_other)
    print(f" WN : [{np.min(Y_other)} ; {np.max(Y_other)}]")
    X_in = X_other[Y_other>0.5][:n_train]
    X_out = X_other[Y_other<0.5][:n_train]
    print(f" Sampled : {X_in.shape[0]} (inside), {X_out.shape[0]} (outside), {X_bd.shape[0]} (surface)")
    return X_bd, N_bd, X_in, X_out

def extract_train_point_cloud_unsigned(n_pt, mesh, domain):
    X_on = M.processing.sampling.sample_points_from_surface(mesh, n_pt)
    domain.pad(0.05,0.05,0.05)
    X_out1 = M.processing.sampling.sample_bounding_box_3D(domain, n_pt//2)
    domain.pad(0.95,0.95,0.95)
    X_out2 = M.processing.sampling.sample_bounding_box_3D(domain, n_pt//2)
    X_out = np.concatenate((X_out1, X_out2))
    print(f"Sampled : {X_on.shape[0]} (surface), {X_out.shape[0]} (outside)")
    return X_on, X_out

def extract_train_point_cloud_distances(n_surf, n_pt, mesh, domain):
    print(" | Sample points on surface")
    X_on = M.processing.sampling.sample_points_from_surface(mesh, n_surf)
    print(" | Sample uniform distribution in domain")
    domain.pad(0.05,0.05,0.05)
    X_out1 = M.processing.sampling.sample_bounding_box_3D(domain, n_pt - n_pt//4)
    domain.pad(1.95,1.95,1.95)
    X_out2 = M.processing.sampling.sample_bounding_box_3D(domain, n_pt//4)
    X_out = np.concatenate((X_out1, X_out2))
    print(" | Compute distances")
    Y_out,_,_ = signed_distance(X_out, np.array(mesh.vertices), np.array(mesh.faces))
    print(f"Sampled : {X_on.shape[0]} (surface), {X_out.shape[0]} (outside)")
    return np.concatenate((X_out,X_on)), np.concatenate((Y_out, np.zeros(X_on.shape[0])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_mesh", type=str, \
        help="path to the input mesh")
    parser.add_argument("-mode", "--mode", default="signed", choices=["signed", "unsigned", "dist"])
    parser.add_argument("-no", "--n-train", type=int, default=100_000)
    parser.add_argument("-ni", "--n-boundary", type=int, default=10_000)
    parser.add_argument("-nt", "--n-test",  type=int, default=10_000)
    parser.add_argument("-nti", "--n-test-boundary", type=int, default=2000)
    parser.add_argument("-visu", help="generates visualization point cloud", action="store_true")
    parser.add_argument("--importance-sampling", help="Importance sampling in the dist mode. Ignored in other modes", action="store_true")
    args = parser.parse_args()

    os.makedirs("inputs", exist_ok=True)

    print("Load mesh")
    mesh = M.mesh.load(args.input_mesh)
    mesh = M.transform.fit_into_unit_cube(mesh)
    mesh = M.transform.translate(mesh, -np.mean(mesh.vertices, axis=0))
    domain = M.geometry.BB3D.of_mesh(mesh)
    
    arrays_to_save = dict()
    mesh_to_save = dict()

    print("Generate train set")
    match args.mode:

        case "unsigned":
            X_on, X_out = extract_train_point_cloud_unsigned(args.n_train, mesh, domain)
            arrays_to_save = {"Xtrain_on" : X_on, "Xtrain_out" : X_out}
            if args.visu:
                mesh_to_save["pts_on"] = point_cloud_from_array(X_on)
                mesh_to_save["pts_train"] = point_cloud_from_arrays((X_on,-1), (X_out, 1.))

        case "signed":
            X_on, N, X_in, X_out = extract_train_point_cloud(args.n_boundary, args.n_train, mesh, domain)
            arrays_to_save = { "Xtrain_on" : X_on, "Xtrain_in" : X_in, "Xtrain_out" : X_out, "Nrml" : N}
            if args.visu:
                mesh_to_save["pts_train"] = point_cloud_from_arrays((X_in,-1),(X_out,1),(X_on,0))
                mesh_to_save["pts_on"] = point_cloud_from_array(X_on)
                mesh_to_save["normals"] = vector_field_from_array(X_on, N, 0.1)

        case "dist":
            if args.importance_sampling:
                far = args.n_train//10
                near = args.n_train - far                
                BETA = 30
                X_on = M.processing.sampling.sample_points_from_surface(mesh, args.n_boundary)
                X_far, Y_far = extract_train_point_cloud_distances(10, far, mesh, domain)
                X_u, Y_u = extract_train_point_cloud_distances(10, 10*args.n_train, mesh, domain)            
                weight = np.exp(-BETA*abs(Y_u))
                weight /= np.sum(weight)
                sampled = np.random.choice(X_u.shape[0],size=far,replace=False,p=weight)
                X_train = np.concatenate((X_far, X_u[sampled],X_on))
                Y_train = np.concatenate((Y_far, Y_u[sampled], np.zeros(args.n_boundary)))
            else:
                X_train, Y_train = extract_train_point_cloud_distances(args.n_boundary, args.n_train, mesh, domain)
            arrays_to_save = {"Xtrain" : X_train, "Ytrain" : Y_train}
            if args.visu:
                mesh_to_save["pts_train"] = point_cloud_from_array(X_train,Y_train)

    print("\nGenerate test set")
    print(" | Sampling points on surface")
    X_surf_test = M.processing.sampling.sample_points_from_surface(mesh, args.n_test_boundary)
    print(" | Sampling uniform distribution in domain")
    X_other_test1 = M.processing.sampling.sample_bounding_box_3D(M.geometry.BB3D.of_mesh(mesh,padding=0.1), args.n_test//2)
    X_other_test2 = M.processing.sampling.sample_bounding_box_3D(M.geometry.BB3D.of_mesh(mesh,padding=1.), args.n_test//2)
    X_other_test = np.concatenate((X_other_test1, X_other_test2))
    
    print(" | Compute distances")
    Y_test,_,_ = signed_distance(X_other_test, np.array(mesh.vertices), np.array(mesh.faces, dtype=np.int32))
    if args.mode == "unsigned":
        Y_test = abs(Y_test)
    X_test = np.concatenate((X_surf_test, X_other_test))
    Y_test = np.concatenate((np.zeros(X_surf_test.shape[0]), Y_test))
    arrays_to_save["Xtest"] = X_test
    arrays_to_save["Ytest"] = Y_test
    if args.visu:
        mesh_to_save["pts_test"] = point_cloud_from_array(X_test,Y_test)


    ### Save generated points
    name = M.utils.get_filename(args.input_mesh)
    if args.visu:
        print("\nGenerate visualization output")
        for file,mesh in mesh_to_save.items():
            M.mesh.save(mesh, f"inputs/{name}_{file}.geogram_ascii")

    print("Saving files")
    for file,ar in arrays_to_save.items():
        np.save(f"inputs/{name}_{file}.npy", ar)
