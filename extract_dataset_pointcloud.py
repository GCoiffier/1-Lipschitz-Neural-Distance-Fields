import mouette as M

import os
import numpy as np
import argparse
import igl
from igl import fast_winding_number_for_points
from common.visualize import point_cloud_from_array, point_cloud_from_arrays, vector_field_from_array

def extract_train_point_cloud(n_pts, V,N, domain):
    domain.pad(0.05,0.05,0.05)
    X_1 = M.sampling.sample_bounding_box_3D(domain, 5*args.n_train)
    domain.pad(0.95,0.95,0.95)
    X_2 = M.sampling.sample_bounding_box_3D(domain, 5*args.n_train)
    X_other = np.concatenate((X_1, X_2))
    WN = fast_winding_number_for_points(V,N, np.ones(V.shape[0]), X_other)
    print("WN:",np.min(WN), np.max(WN))
    X_out = X_other[WN>0.5]
    X_in = X_other[WN<0.5]
    np.random.shuffle(X_out)
    np.random.shuffle(X_in)
    X_out = X_out[:n_pts]
    X_in = X_in[:n_pts]
    return X_out,X_in

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="path to the input point cloud")
    parser.add_argument("-mode", "--mode", default="signed", choices=["signed", "unsigned"])
    parser.add_argument("-no", "--n-train", type=int, default=100_000)
    parser.add_argument("-visu", help="generates visualization point cloud", action="store_true")
    args = parser.parse_args()

    os.makedirs("inputs", exist_ok=True)

    print("Load mesh")
    pc = M.mesh.load(args.input)
    pc = M.transform.fit_into_unit_cube(pc)
    pc = M.transform.translate(pc, -np.mean(pc.vertices, axis=0))
    domain = M.geometry.BB3D.of_mesh(pc)
    Vtx = np.array(pc.vertices)

    if pc.vertices.has_attribute("normals"):
        Nrml = pc.vertices.get_attribute("normals").as_array(len(pc.vertices))
    else:
        # TODO: implement some normal estimation for point clouds in Mouette
        raise NotImplementedError

    print("Generate train set")
    mesh_to_save = dict()
    arrays_to_save = {
        "X_on" : Vtx,
        "Nrml" : Nrml 
    }
    match args.mode:
        case "unsigned":
            domain.pad(0.05,0.05,0.05)
            X_out1 = M.sampling.sample_bounding_box_3D(domain, args.n_train//2)
            domain.pad(0.95,0.95,0.95)
            X_out2 = M.sampling.sample_bounding_box_3D(domain, args.n_train//2)
            X_out = np.concatenate((X_out1, X_out2))
            arrays_to_save["X_out"] = X_out
            if args.visu:
                mesh_to_save["pts"] = point_cloud_from_array((Vtx,-1), (X_out,1.))
                mesh_to_save["normals"] = vector_field_from_array(Vtx, Nrml, 0.1)
        case "signed":
            X_in, X_out = extract_train_point_cloud(args.n_train, Vtx, Nrml, domain)
            arrays_to_save["X_in"] = X_in
            arrays_to_save["X_out"] = X_out
            if args.visu:
                mesh_to_save["pts_on"] = point_cloud_from_array(Vtx)
                mesh_to_save["pts_in"] = point_cloud_from_array(X_in)
                mesh_to_save["pts_train"] = point_cloud_from_arrays((X_in,-1), (Vtx,0.), (X_out,1.))
                mesh_to_save["normals"] = vector_field_from_array(Vtx, Nrml, 0.1)

    name = M.utils.get_filename(args.input)
    if args.visu:
        print("\nGenerate visualization output")
        for file,mesh in mesh_to_save.items():
            M.mesh.save(mesh, f"inputs/{name}_{file}.geogram_ascii")

    print("Saving files")
    for file,ar in arrays_to_save.items():
        np.save(f"inputs/{name}_{file}.npy", ar)