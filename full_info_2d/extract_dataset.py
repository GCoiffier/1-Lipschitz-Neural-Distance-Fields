import mouette as M
from mouette import geometry as geom

import numpy as np
import argparse
from numba import jit, prange
from tqdm import tqdm

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

#@jit(cache=True, nopython=True, parallel=True)
def compute_Y(X, PL):
    """
    Args:
        X : query points array
        PL (_type_): _description_

    Returns:
        _type_: _description_
    """
    nX = X.shape[0]
    Y = np.zeros(nX)
    for iX in tqdm(range(nX)):
        Y[iX] = signed_distance(X[iX,:], PL)
    return Y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_mesh", type=str, \
        help="path to the input mesh")

    parser.add_argument("-n", "--n-train", type=int, default=10000, \
        help="number of sample points in train set")

    parser.add_argument("-m", "--n-test", type=int, default=1000, \
        help="number of sample points in test set")

    parser.add_argument("-visu", help="generates visualization point cloud", action="store_true")

    args = parser.parse_args()

    print("Load mesh")
    full_mesh = M.mesh.load(args.input_mesh)

    print("Extract boundary polyline")
    boundary, _ = M.processing.extract_curve_boundary(full_mesh)
    boundary = M.transform.fit_into_unit_cube(boundary)

    b_edges = np.zeros((len(boundary.edges), 2, 2))
    for i,(A,B) in enumerate(boundary.edges): 
        pA,pB = boundary.vertices[A], boundary.vertices[B]
        pA = (pA.x, pA.y)
        pB = (pB.x, pB.y)
        b_edges[i,0,:] = pA
        b_edges[i,1,:] = pB

    print("Generate train set")
    X_train = np.array([-0.2,-0.2]) + 1.4*np.random.random((args.n_train, 2))
    Y_train = compute_Y(X_train, b_edges)

    print("Generate test set")
    X_test = np.array([-0.2,-0.2]) + 1.4*np.random.random((args.n_test, 2))
    Y_test = compute_Y(X_test, b_edges)
    
    print("Removing points with large distance")
    ymin = min( np.amin(Y_train), np.amin(Y_test))
    print(f"Min distance: {ymin}")
    X_train, Y_train, X_test, Y_test = X_train[Y_train<-ymin], Y_train[Y_train<-ymin], X_test[Y_test<-ymin], Y_test[Y_test<-ymin]

    if args.visu:
        print("Generate visualization output")
        pt_cloud = M.mesh.PointCloud()
        test_train_attr = pt_cloud.vertices.create_attribute("test", bool)
        dist_attr = pt_cloud.vertices.create_attribute("d", float)
        for i in range(X_train.shape[0]):
            pt_cloud.vertices.append(geom.Vec(X_train[i,0], X_train[i,1], 0.))
            dist_attr[i] = Y_train[i]

        n = X_train.shape[0]
        for i in range(X_test.shape[0]):
            pt_cloud.vertices.append(geom.Vec(X_test[i,0], X_test[i,1], 0.))
            dist_attr[n+i] = Y_test[i]
            test_train_attr[n+1] = True

    print("Saving files")
    name = M.utils.get_filename(args.input_mesh)
    if args.visu:
        M.mesh.save(pt_cloud, f"inputs/{name}_pts.geogram_ascii")
        M.mesh.save(boundary, f"inputs/{name}_boundary.mesh")
    np.save(f"inputs/{name}_Xtrain.npy", X_train)
    np.save(f"inputs/{name}_Ytrain.npy", Y_train)
    np.save(f"inputs/{name}_Xtest.npy", X_test)
    np.save(f"inputs/{name}_Ytest.npy", Y_test)