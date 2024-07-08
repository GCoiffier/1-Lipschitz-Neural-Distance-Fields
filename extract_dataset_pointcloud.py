import mouette as M

import os
import numpy as np
import argparse
from collections import deque
from tqdm import trange

from igl import fast_winding_number_for_points
from scipy.spatial import KDTree
import triangle

from common.visualize import point_cloud_from_array, point_cloud_from_arrays, vector_field_from_array

def extract_train_point_cloud(n_pts, V,N,A, domain, t_in=0., t_out=0.):
    """Sample a point cloud in a given domain and partition it using the generalized winding number.
    Half the points are taken in a tighter domain around the point cloud.

    Args:
        n_pts (int): number of points to sample
        V (np.ndarray): vertices of the point cloud (shape Nx3)
        N (np.ndarray): normal vectors per vertex (shape Nx3)
        A (np.ndarray): local area per vertex (shape Nx1)
        domain (M.geometry.AABB): axis-aligned domain inside which points are uniformly sampled
        t_in (float, optional): GWN threshold for inside points. Samples are kept if GWN >= t_in. Defaults to 0..
        t_out (float, optional): GWN threshold for outside points. Samples are kept if GWN <= t_out. Defaults to 0..

    Returns:
        X_in, X_out: two arrays of 'n_pts/2' points that are inside and outside the input point cloud.
    """
    t_in = max(0., t_in)
    domain.pad(0.05)
    X_1 = M.sampling.sample_AABB(domain, 20*args.n_train)
    domain.pad(0.95)
    X_2 = M.sampling.sample_AABB(domain, args.n_train//2)
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

def estimate_normals(V, tree:KDTree, k=30):
    """Estimation of outward normals for each vertex of the point cloud. Computed as a linear regression on k nearest neighbors

    Args:
        V (np.ndarray): array of vertex positions (Nx3)
        tree (KDTree): kdtree for fast neighbor queries
        k (int, optional): number of neighbor vertices to take into account for computation. Defaults to 30.

    Returns:
        np.ndarray: array of estimated normals (Nx3)
    """
    n_pts = V.shape[0]
    _, KNN = tree.query(V, k+1)
    KNN = KNN[:,1:] # remove self from nearest
    
    # Compute normals
    N = np.zeros((n_pts, 3))
    for i in trange(n_pts):
        mat_i = np.array([V[nn,:]-V[i,:] for nn in KNN[i,:]])
        mat_i = mat_i.T @ mat_i
        _, eigs = np.linalg.eigh(mat_i)
        N[i,:] = eigs[:,0]

    # Consistent normal orientation
    visited = np.zeros(n_pts,dtype=bool)
    to_visit = deque()
    exterior_pt = M.Vec(10,0,0)
    _,top_pt = tree.query(exterior_pt)
    if np.dot(N[i,:], exterior_pt - V[top_pt,:])<0.:
        N[i,:] *= -1 # if this normal is not outward, we are doomed
    to_visit.append(top_pt)
    while len(to_visit)>0:
        iv = to_visit.popleft()
        if visited[iv] : continue
        visited[iv] = True
        Nv = N[iv, :]
        for nn in KNN[iv,:]:
            if np.dot(Nv, N[nn,:])<0.:
                N[nn,:] *= -1
            to_visit.append(nn)
    return N

def estimate_vertex_areas(V,N, tree:KDTree, k=10):
    """The generalized winding number needs an estimate of 'local areas' of each points. This functions computes this area estimation as described in the article of Barill et al. (2018).

    Args:
        V (np.ndarray): array of vertex positions (Nx3)
        N (np.ndarray): array of normal vector per vertex (Nx3)
        tree (KDTree): kdtree for fast neighbor queries
        k (int, optional): number of neighbor vertices to take into account for computation. Defaults to 20.

    Returns:
        np.ndarray: array of estimated local areas (Nx1)
    """
    n_pts = V.shape[0]
    _, KNN = tree.query(V,k+1)
    KNN = KNN[:,1:]
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
    parser = argparse.ArgumentParser(
        prog="Dataset Generator",
        description="Generate a dataset to train a neural distance field from a point cloud with normals"
    )

    parser.add_argument("input", type=str, help="path to the input point cloud")
    parser.add_argument("-mode", "--mode", default="signed", choices=["signed", "unsigned"], help="dataset mode. 'signed' runs the generalized winding number to label points as inside/outside. 'unsigned' labels points as boundary/else.")
    parser.add_argument("-no", "--n-train", type=int, default=100_000, help="number of samples in the training set")
    parser.add_argument("-visu", help="generates visualization point cloud", action="store_true")
    parser.add_argument("-ti", "--threshold-in", type=float, default=0.5, help="keep interior if WN is > 0.5+t. Ignored in unsigned mode")
    parser.add_argument("-to", "--threshold-out", type=float, default=0., help="keep exterior if WN is < 0.5-t. Ignored in unsigned mode")
    args = parser.parse_args()

    os.makedirs("inputs", exist_ok=True)

    print("Load mesh")
    pc = M.mesh.load(args.input)

    pc = M.transform.fit_into_unit_cube(pc)
    pc = M.transform.translate(pc, -np.mean(pc.vertices, axis=0))
    domain = M.geometry.AABB.of_mesh(pc)
    Vtx = np.array(pc.vertices)

    if not pc.vertices.has_attribute("normals") or not pc.vertices.has_attribute("area"):
        # precompute kdtree once
        kdtree = KDTree(pc.vertices)

    if pc.vertices.has_attribute("normals"):
        N = pc.vertices.get_attribute("normals").as_array(len(pc.vertices))
    else:
        print("Estimate normals")
        N = estimate_normals(Vtx, kdtree)
        pc.vertices.register_array_as_attribute("normals", N)
        if args.visu:
            poly_normals = vector_field_from_array(Vtx, N, 0.01)
            M.mesh.save(poly_normals, f"inputs/normals.mesh")

    if pc.vertices.has_attribute("area"):
        A = pc.vertices.get_attribute("area").as_array(len(pc.vertices))
    else:
        print("Estimate vertex local area")
        A = estimate_vertex_areas(Vtx, N, kdtree)
        pc.vertices.register_array_as_attribute("area", A)

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
            domain.pad(0.05)
            X_out1 = M.sampling.sample_AABB(domain, n_pts_tight)
            domain.pad(0.95)
            X_out2 = M.sampling.sample_AABB(domain, n_pts_large)
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
