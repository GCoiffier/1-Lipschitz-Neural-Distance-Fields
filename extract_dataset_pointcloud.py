import mouette as M

import os
import numpy as np
import argparse
from tqdm import trange

from igl import fast_winding_number_for_points
from sklearn.neighbors import kneighbors_graph
import triangle

from common.visualize import point_cloud_from_array, point_cloud_from_arrays, vector_field_from_array

def extract_train_point_cloud(n_pts, V,N,A, domain, t_in=0., t_out=0.):
    t_in = max(0., t_in)
    domain.pad(0.05,0.05,0.05)
    X_1 = M.sampling.sample_bounding_box_3D(domain, 20*args.n_train)
    domain.pad(0.95,0.95,0.95)
    X_2 = M.sampling.sample_bounding_box_3D(domain, args.n_train//2)
    X_other = np.concatenate((X_1, X_2))
    WN = fast_winding_number_for_points(V,N,A, X_other)
    print("WN:",np.min(WN), np.max(WN))
    X_out = X_other[WN<0.5-t_out]
    X_in = X_other[WN>0.5+t_in]
    np.random.shuffle(X_out)
    np.random.shuffle(X_in)
    X_out = X_out[:n_pts]
    X_in = X_in[:n_pts]
    return X_in, X_out

def estimate_normals(V):
    raise NotImplementedError

def estimate_vertex_areas(V,N, k=20):
    n_pts = V.shape[0]
    KNN_mat = kneighbors_graph(V,k,mode="connectivity")
    KNN = [[] for _ in range(n_pts)]
    rows, cols = KNN_mat.nonzero()
    for r,c in zip(rows,cols):
        KNN[r].append(c)
    KNN = np.array(KNN)
    A = np.zeros(n_pts)
    for i in trange(n_pts):
        ni = M.Vec.normalized(N[i])
        Xi,Yi,Zi = (M.geometry.cross(basis, ni) for basis in (M.Vec.X(), M.Vec.Y(), M.Vec.Z()))
        Xi = [_X for _X in (Xi,Yi,Zi) if M.geometry.norm(_X)>1e-8][0]
        Xi = M.Vec.normalized(Xi)
        Yi = M.geometry.cross(ni,Xi)

        neighbors = [V[j] for j in KNN[i,:]] # coordinates of k neighbors
        neighbors = [M.geometry.project_to_plane(_X,N[i],V[i]) for _X in neighbors] # Project onto normal plane
        neighbors = [M.Vec(Xi.dot(_X), Yi.dot(_X)) for _X in neighbors]

        for (p1,p2,p3) in triangle.triangulate({"vertices" : neighbors})["triangles"]:
            A[i] += M.geometry.triangle_area_2D(neighbors[p1], neighbors[p2], neighbors[p3])
    return A

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="path to the input point cloud")
    parser.add_argument("-mode", "--mode", default="signed", choices=["signed", "unsigned"])
    parser.add_argument("-no", "--n-train", type=int, default=100_000)
    parser.add_argument("-visu", help="generates visualization point cloud", action="store_true")
    parser.add_argument("-ti", "--threshold-in", type=float, default=0., help="keep interior if WN is > 0.5+t. Ignored in unsigned mode")
    parser.add_argument("-to", "--threshold-out", type=float, default=0., help="keep exterior if WN is < 0.5-t. Ignored in unsigned mode")
    args = parser.parse_args()

    os.makedirs("inputs", exist_ok=True)

    print("Load mesh")
    pc = M.mesh.load(args.input)
    pc = M.transform.fit_into_unit_cube(pc)
    pc = M.transform.translate(pc, -np.mean(pc.vertices, axis=0))
    domain = M.geometry.BB3D.of_mesh(pc)
    Vtx = np.array(pc.vertices)

    if pc.vertices.has_attribute("normals"):
        N = pc.vertices.get_attribute("normals").as_array(len(pc.vertices))
    else:
        print("Estimate normals")
        N = estimate_normals(Vtx)
        N_attr = pc.vertices.get_attribute("normals", float, 3, dense=True)
    
    if pc.vertices.has_attribute("area"):
        A = pc.vertices.get_attribute("area").as_array(len(pc.vertices))
    else:
        print("Estimate vertex local area")
        A = estimate_vertex_areas(Vtx,N)
        A_attr = pc.vertices.create_attribute("area", float, dense=True)
        A_attr._data = A[:,np.newaxis]

    if args.visu:
        file_name = M.utils.get_filename(args.input)
        M.mesh.save(pc, f"inputs/{file_name}.geogram_ascii")

    print("Generate train set")
    mesh_to_save = dict()
    arrays_to_save = {
        "Xtrain_on" : Vtx,
    }
    match args.mode:
        case "unsigned":
            n_pts = Vtx.shape[0]
            n_pts_large = n_pts//4
            n_pts_tight = n_pts - n_pts_large
            domain.pad(0.05,0.05,0.05)
            X_out1 = M.sampling.sample_bounding_box_3D(domain, n_pts_tight)
            domain.pad(0.95,0.95,0.95)
            X_out2 = M.sampling.sample_bounding_box_3D(domain, n_pts_large)
            X_out = np.concatenate((X_out1, X_out2))
            arrays_to_save["Xtrain_out"] = X_out
            if args.visu:
                mesh_to_save["pts"] = point_cloud_from_arrays((Vtx,-1), (X_out,1.))
                mesh_to_save["normals"] = vector_field_from_array(Vtx, N, 0.02)
        case "signed":
            X_in, X_out = extract_train_point_cloud(args.n_train, Vtx, N, A, domain, t_in=args.threshold_in, t_out=args.threshold_out)
            arrays_to_save["Xtrain_in"] = X_in
            arrays_to_save["Xtrain_out"] = X_out
            if args.visu:
                mesh_to_save["pts_on"] = point_cloud_from_array(Vtx)
                mesh_to_save["pts_in"] = point_cloud_from_array(X_in)
                mesh_to_save["pts_train"] = point_cloud_from_arrays((X_in,-1), (Vtx,0.), (X_out,1.))
                mesh_to_save["normals"] = vector_field_from_array(Vtx, N, 0.1)

    name = M.utils.get_filename(args.input)
    if args.visu:
        print("\nGenerate visualization output")
        for file,mesh in mesh_to_save.items():
            M.mesh.save(mesh, f"inputs/{name}_{file}.geogram_ascii")

    print("Saving files")
    for file,ar in arrays_to_save.items():
        np.save(f"inputs/{name}_{file}.npy", ar)
