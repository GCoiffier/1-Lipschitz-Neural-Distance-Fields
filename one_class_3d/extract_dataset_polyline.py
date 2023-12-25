import mouette as M
from mouette import geometry as geom

import os
import numpy as np
import argparse
from numba import jit, prange

@jit(cache=True, nopython=True)
def norm(A):
    return np.sqrt(np.dot(A,A)) 

@jit(cache=True, nopython=True, parallel=True)
def distance(P, PL):
    nE = PL.shape[0]
    d = 100.
    for iE in prange(nE):
        pA,pB = PL[iE,0,:], PL[iE,1,:]
        d = min(d, norm(P-pA))
        d = min(d, norm(P-pB))
    return d

@jit(cache=True, nopython=True, parallel=True)
def compute_Y(X, PL):
    """
    Args:
        X : query points array
        PL : polyline
    """
    nX = X.shape[0]
    Y = np.zeros(nX)
    for iX in prange(nX):
        Y[iX] = distance(X[iX,:], PL)
    return Y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_mesh", type=str, \
        help="path to the input mesh")

    parser.add_argument("-n", "--n-train", type=int, default=5000, \
        help="number of sample points in train set")

    parser.add_argument("-m", "--n-test", type=int, default=1000, \
        help="number of sample points in test set")
    
    parser.add_argument("-visu", help="generates visualization point cloud", action="store_true")

    args = parser.parse_args()

    os.makedirs("inputs", exist_ok=True)

    print("Load mesh")
    full_mesh = M.mesh.load(args.input_mesh)
    
    print("Extract boundary polyline")
    boundary, _ = M.processing.extract_curve_boundary(full_mesh)
    boundary = M.transform.fit_into_unit_cube(boundary)
    domain = M.geometry.BB3D.of_mesh(boundary, padding=0.5)

    b_edges = np.zeros((len(boundary.edges), 2, 3))
    for i,(A,B) in enumerate(boundary.edges): 
        pA,pB = boundary.vertices[A], boundary.vertices[B]
        b_edges[i,0,:] = pA
        b_edges[i,1,:] = pB

    print("Generate train set")
    X_train = M.processing.sampling.sample_points_from_polyline(boundary, args.n_train)
    print(X_train.shape)

    print("Generate test set")
    X_test = M.processing.sampling.sample_bounding_box_3D(domain, args.n_test)
    Y_test = compute_Y(X_test, b_edges)

    print("Removing points with large distance")
    ymin = np.amin(Y_test)-1.
    print(f"Min distance: {ymin}")
    X_test, Y_test = X_test[Y_test<-ymin], Y_test[Y_test<-ymin]

    if args.visu:
        print("Generate visualization output")
        pt_cloud = M.mesh.PointCloud()
        test_train_attr = pt_cloud.vertices.create_attribute("test", bool)
        dist_attr = pt_cloud.vertices.create_attribute("d", float)
        for i in range(X_train.shape[0]):
            pt_cloud.vertices.append(X_train[i])

        n = X_train.shape[0]
        for i in range(X_test.shape[0]):
            pt_cloud.vertices.append(X_test[i])
            dist_attr[n+i] = Y_test[i]
            test_train_attr[n+1] = True

    print("Saving files")
    name = M.utils.get_filename(args.input_mesh)
    if args.visu:
        M.mesh.save(pt_cloud, f"inputs/{name}_pts.geogram_ascii")
        M.mesh.save(boundary, f"inputs/{name}_boundary.mesh")
    np.save(f"inputs/{name}_Xtrain.npy", X_train)
    np.save(f"inputs/{name}_Xtest.npy", X_test)
    np.save(f"inputs/{name}_Ytest.npy", Y_test)