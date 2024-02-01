import mouette as M
from mouette import geometry as geom

import os
import numpy as np
from numpy.random import choice
import argparse
from numba import jit, prange

@jit(cache=True, nopython=True)
def cross(A,B):
    return A[0]*B[1] - A[1]*B[0]

@jit(cache=True, nopython=True)
def distance_to_segment2D(P, A, B):
    """
    Computes the distance of point P to the segment [A;B]
    """
    P,A,B = P[:2], A[:2], B[:2]
    seg = B-A
    seg_length_sq = np.dot(seg, seg)
    if seg_length_sq<1e-12: 
        # segment is a single point
        return np.sqrt(np.dot(P-A,P-A))
    t = max(0., min(1., np.dot(P-A, seg)/seg_length_sq))
    proj = A + t*seg
    return np.sqrt(np.dot(P-proj, P-proj))

@jit(cache=True, nopython=True)
def intersect_ray_segment2D(P, A, B):
    r = np.array([1.,0.])
    s = B-A
    rs = cross(r,s)
    aps = cross(P-A, s)
    if abs(rs)<1e-10: # ray is parallel to segment
        return abs(aps)<1e-10 # segment is confounded in ray
    t = aps / rs
    u = cross(A-P, r)/rs
    return (t>=0. and 0<=u<=1)

@jit(cache=True, nopython=True, parallel=True)
def signed_distance(P, PL):
    nE = PL.shape[0]
    n_collisions = 0
    d = 100.
    for iE in prange(nE):
        pA,pB = PL[iE,0,:], PL[iE,1,:]
        if intersect_ray_segment2D(P, pA,pB): 
            n_collisions = n_collisions + 1
        d = min(d, distance_to_segment2D(P,pA,pB))
    if n_collisions % 2 == 1:
        return -d
    return d

@jit(cache=True, nopython=True, parallel=True)
def compute_distances(Q, PL):
    """
    Args:
        Q : query points array
        PL : polyline
    """
    nQ = Q.shape[0]
    Y = np.zeros(nQ)
    for i in prange(nQ):
        Y[i] = signed_distance(Q[i,:], PL)
    return Y

@jit(cache=True, nopython=True, parallel=True)
def count_ray_parity(P, PL):
    nE = PL.shape[0]
    n_collisions = 0
    for iE in prange(nE):
        pA,pB = PL[iE,0,:], PL[iE,1,:]
        if intersect_ray_segment2D(P, pA,pB): 
            n_collisions = n_collisions + 1
    return n_collisions%2

@jit(cache=True, nopython=True, parallel=True)
def compute_inside_outside(Q, PL):
    """
    Args:
        Q : query points array
        PL : polyline
    """
    nQ = Q.shape[0]
    Y = np.zeros(nQ)
    for i in prange(nQ):
        Y[i] = count_ray_parity(Q[i,:], PL)
    return Y

def sample_points_and_normals(mesh, n_pts):
    sampled_pts = np.zeros((n_pts, 2))
    lengths = M.attributes.edge_length(mesh, persistent=False).as_array()
    lengths /= np.sum(lengths)
    if len(mesh.edges)==1:
        edges = [0]*n_pts
    else:
        edges = choice(len(mesh.edges), size=n_pts, p=lengths)
    sampled_normals = np.zeros((n_pts,2))
    for i,e in enumerate(edges):
        pA,pB = (mesh.vertices[_v] for _v in mesh.edges[e])
        ni = M.Vec.normalized(pB - pA)
        sampled_normals[i,:] = np.array([ni.y, -ni.x])
        t = np.random.random()
        pt = t*pA + (1-t)*pB
        sampled_pts[i,:] = pt[:2] 
    return sampled_pts, sampled_normals

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_mesh", type=str, \
        help="path to the input mesh")
    parser.add_argument("--unsigned", action="store_true")
    parser.add_argument("-no", "--n-train", type=int, default=10000)
    parser.add_argument("-ni", "--n-boundary", type=int, default=5000)
    parser.add_argument("-nt", "--n-test",  type=int, default=3000)
    parser.add_argument("-nti", "--n-test-boundary", type=int, default=1000)
    parser.add_argument("-visu", help="generates visualization point cloud", action="store_true")
    args = parser.parse_args()

    os.makedirs("inputs", exist_ok=True)

    print("Load mesh")
    input_mesh = M.mesh.load(args.input_mesh)
    input_mesh = M.transform.fit_into_unit_cube(input_mesh)
    input_mesh = M.transform.flatten(input_mesh, dim=2) # make sure that z = 0
    input_mesh = M.transform.translate_to_origin(input_mesh)

    if isinstance(input_mesh, M.mesh.SurfaceMesh):
        print("Extract boundary polyline")
        mesh, _ = M.processing.extract_curve_boundary(input_mesh)
    elif isinstance(input_mesh, M.mesh.PolyLine):
        mesh = input_mesh

    domain = M.geometry.BB2D.of_mesh(mesh, padding=0.5)
    b_edges = np.zeros((len(mesh.edges), 2, 2))
    for i,(A,B) in enumerate(mesh.edges): 
        pA,pB = mesh.vertices[A], mesh.vertices[B]
        pA = (pA.x, pA.y)
        pB = (pB.x, pB.y)
        b_edges[i,0,:] = pA
        b_edges[i,1,:] = pB

    print("Generate train set")
    print(" | Sampling points")
    X_bd, N_bd = sample_points_and_normals(mesh, args.n_boundary)
    mult = 20
    ok = False
    while not ok:
        X_other = M.processing.sampling.sample_bounding_box_2D(domain, mult*args.n_train)[:,:2]
        Y_other = compute_distances(X_other, b_edges)
        X_in = X_other[Y_other<-1e-3, :][:args.n_train]
        X_out = X_other[Y_other>1e-2, :][:args.n_train]
        ok = (X_in.shape[0] == args.n_train and X_out.shape[0] == args.n_train)
        mult *= 2
    print(f" | Generated {X_in.shape[0]} (inside), {X_out.shape[0]} (outside), {X_bd.shape[0]} (boundary)")

    print("Generate test set")
    args.n_test_boundary = min(args.n_test_boundary, args.n_boundary)
    X_test = M.processing.sampling.sample_bounding_box_2D(domain, args.n_test)
    Y_test = compute_distances(X_test, b_edges)
    X_test = np.concatenate((X_test, X_bd[np.random.choice(X_bd.shape[0], args.n_test_boundary, replace=False), :]))
    Y_test = np.concatenate((Y_test,np.zeros(args.n_test_boundary)))
    
    if args.visu:
        print("Generate visualization output")
        pc_train = M.mesh.PointCloud()
        in_out_attr = pc_train.vertices.create_attribute("in", int)
        for i in range(X_in.shape[0]):
            pc_train.vertices.append(geom.Vec(X_in[i,0], X_in[i,1], 0.))
            in_out_attr[i] = -1
        n = len(pc_train.vertices)
        for i in range(X_bd.shape[0]):
            pc_train.vertices.append(geom.Vec(X_bd[i,0], X_bd[i,1], 0.))
            in_out_attr[n+i] = 0
        n = len(pc_train.vertices)
        for i in range(X_out.shape[0]):
            pc_train.vertices.append(geom.Vec(X_out[i,0], X_out[i,1], 0.))
            in_out_attr[n+i] = 1

        pc_test  = M.mesh.PointCloud()
        dist_attr = pc_test.vertices.create_attribute("d", float)
        for i in range(X_test.shape[0]):
            pc_test.vertices.append(geom.Vec(X_test[i,0], X_test[i,1], 0.))
            dist_attr[i] = Y_test[i]

        nrml_poly = M.mesh.PolyLine()
        for i in range(X_bd.shape[0]):
            p1 = geom.Vec(X_bd[i,0], X_bd[i,1], 0.)
            p2 = p1 + 0.1*M.Vec(N_bd[i,0], N_bd[i,1], 0.)
            nrml_poly.vertices += [p1, p2]
            nrml_poly.edges.append((2*i, 2*i+1))

    print("Saving files")
    name = M.utils.get_filename(args.input_mesh)
    if args.visu:
        M.mesh.save(pc_train, f"inputs/{name}_pts_train.geogram_ascii")
        M.mesh.save(pc_test, f"inputs/{name}_pts_test.geogram_ascii")
        M.mesh.save(nrml_poly,f"inputs/{name}_normals.mesh")
    np.save(f"inputs/{name}_Xtrain_in.npy", X_in)
    np.save(f"inputs/{name}_Xtrain_out.npy", X_out)
    np.save(f"inputs/{name}_Xtrain_bd.npy", X_bd)
    np.save(f"inputs/{name}_Normals_bd.npy", N_bd)
    np.save(f"inputs/{name}_Xtest.npy", X_test)
    np.save(f"inputs/{name}_Ytest.npy", Y_test)