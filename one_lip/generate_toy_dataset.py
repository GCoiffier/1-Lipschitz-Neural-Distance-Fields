import mouette as M
from mouette import geometry as geom

import os
import numpy as np
import argparse
from numba import jit, prange
from tqdm import tqdm
import cmath

def generate_square(n_out, n_surf, n_test):
    SIDE = 0.8
    square = M.geometry.BB2D(- SIDE/2, -SIDE/2, SIDE/2, SIDE/2)
    square_mesh = M.mesh.RawMeshData()
    square_mesh.vertices += [
        [square.left, square.bottom, 0.],
        [square.right, square.bottom, 0.],
        [square.right, square.top, 0.],
        [square.left, square.top, 0.]
    ]
    square_mesh.edges += [(0,1), (0,3), (1,2), (2,3)]
    square_mesh = M.mesh.PolyLine(square_mesh)

    def distance_to_square(x):
        d = np.array(abs(x) - np.array([SIDE/2,SIDE/2]))
        return np.linalg.norm(np.maximum(d,0.)) + min(max(d[0], d[1]), 0.)

    print("Generate train set")
    domain = M.geometry.BB2D(-1.,-1.,1.,1.)
    print(" | Sampling points")
    X_bd = M.processing.sampling.sample_points_from_polyline(square_mesh, n_surf)[:,:2]
    X_other = M.processing.sampling.sample_bounding_box_2D(domain, 10*n_out)[:,:2]
    print(" | Discriminate interior from exterior points")
    D = np.array([distance_to_square(x) for x in X_other])
    X_in = X_other[D<0, :][:n_out]
    X_out = X_other[D>0, :]
    X_out = X_out[:X_in.shape[0], :] # same number of points inside and outside
    print(f" | Generated {X_in.shape[0]} (inside), {X_out.shape[0]} (outside), {X_bd.shape[0]} (boundary)")

    print("Generate test set")
    n_test_surf  = min(n_test//4, X_bd.shape[0])
    n_test_other = n_test - n_test_surf

    X_test = M.processing.sampling.sample_bounding_box_2D(domain, n_test_other)
    Y_test = np.array([distance_to_square(x) for x in X_test])
    X_test = np.concatenate((X_test, X_bd[np.random.choice(X_bd.shape[0], n_test_surf, replace=False), :]))
    Y_test = np.concatenate((Y_test,np.zeros(n_test_surf)))
    return X_in,X_out,X_bd,X_test,Y_test


def generate_square_no_interior(n_train,n_test):
    SIDE = 0.8
    square = M.geometry.BB2D(- SIDE/2, -SIDE/2, SIDE/2, SIDE/2)
    square_mesh = M.mesh.RawMeshData()
    square_mesh.vertices += [
        [square.left, square.bottom, 0.],
        [square.right, square.bottom, 0.],
        [square.right, square.top, 0.],
        [square.left, square.top, 0.]
    ]
    square_mesh.edges += [(0,1), (0,3), (1,2), (2,3)]
    square_mesh = M.mesh.PolyLine(square_mesh)

    print("Generate train set")
    domain = M.geometry.BB2D(-1.,-1.,1.,1.)
    print(" | Sampling points")
    X_on = M.processing.sampling.sample_points_from_polyline(square_mesh, n_train)[:,:2]
    X_out = M.processing.sampling.sample_bounding_box_2D(domain, n_train)[:,:2]
    print(f" | Generated {X_on.shape[0]} (on), {X_out.shape[0]} (outside)")

    print("Generate test set")
    def distance_to_square(x):
        d = np.array(abs(x) - np.array([SIDE/2,SIDE/2]))
        return abs(np.linalg.norm(np.maximum(d,0.)) + min(max(d[0], d[1]), 0.))
    m_surf  = min(n_test//4, X_on.shape[0])
    m_other = n_test - m_surf

    X_test = M.processing.sampling.sample_bounding_box_2D(domain, m_other)
    Y_test = np.array([distance_to_square(x) for x in X_test])
    X_test = np.concatenate((X_test, X_on[np.random.choice(X_on.shape[0], m_surf, replace=False), :]))
    Y_test = np.concatenate((Y_test,np.zeros(m_surf)))
    return X_on,X_out,X_test,Y_test


def generate_circle(n_out, n_surf, n_test):
    RADIUS = 0.4
    distance_to_circle = lambda x : np.linalg.norm(x) - RADIUS

    print("Generate train set")
    domain = M.geometry.BB2D(-1.,-1.,1.,1.)

    print(" | Sampling points")
    angles = np.random.random(n_surf)*2*np.pi
    X_bd = [cmath.rect(RADIUS,ang) for ang in angles]
    X_bd = np.array([[x.real, x.imag] for x in X_bd]) # convert from complexes to vec2
    X_other = M.processing.sampling.sample_bounding_box_2D(domain, 10*n_out)[:,:2]

    print(" | Discriminate interior from exterior points")    
    D = np.array([distance_to_circle(x) for x in X_other])
    X_in = X_other[D<0, :][:n_out]
    X_out = X_other[D>0, :]
    X_out = X_out[:X_in.shape[0], :] # same number of points inside and outside
    print(f" | Generated {X_in.shape[0]} (inside), {X_out.shape[0]} (outside), {X_bd.shape[0]} (boundary)")

    print("Generate test set")
    n_test_surf  = min(n_test//4, X_bd.shape[0])
    n_test_other = n_test - n_test_surf

    X_test = M.processing.sampling.sample_bounding_box_2D(domain, n_test_other)
    Y_test = np.array([distance_to_circle(x) for x in X_test])

    angles = np.random.random(n_surf)*2*np.pi
    X_bd = [cmath.rect(RADIUS,ang) for ang in angles]
    X_bd = np.array([[x.real, x.imag] for x in X_bd]) # convert from complexes to vec2

    X_test = np.concatenate((X_test, X_bd[np.random.choice(X_bd.shape[0], n_test_surf, replace=False), :]))
    Y_test = np.concatenate((Y_test,np.zeros(n_test_surf)))
    return X_in,X_out,X_bd,X_test,Y_test


def generate_circle_no_interior(n_train, n_test):
    RADIUS = 0.4

    print("Generate train set")
    domain = M.geometry.BB2D(-1.,-1.,1.,1.)

    print(" | Sampling points")
    angles = np.random.random(n_train)*2*np.pi
    X_on = [cmath.rect(RADIUS,ang) for ang in angles]
    X_on = np.array([[x.real, x.imag] for x in X_on]) # convert from complexes to vec2
    X_out = M.processing.sampling.sample_bounding_box_2D(domain, n_train)[:,:2]
    print(f" | Generated {X_on.shape[0]} (on), {X_out.shape[0]} (outside)")

    print("Generate test set")
    distance_to_circle = lambda x : abs(np.linalg.norm(x) - RADIUS)
    n_test_surf  = min(n_test//4, X_on.shape[0])
    n_test_other = n_test - n_test_surf
    X_test = M.processing.sampling.sample_bounding_box_2D(domain, n_test_other)
    Y_test = np.array([distance_to_circle(x) for x in X_test])
    X_test = np.concatenate((X_test, X_on[np.random.choice(X_on.shape[0], n_test_surf, replace=False), :]))
    Y_test = np.concatenate((Y_test,np.zeros(n_test_surf)))
    return X_on,X_out,X_test,Y_test


def generate_segment(n_train,n_test):
    A = np.array([-0.5,-0.2])
    B = np.array([0.5,0.2])
    print("Generate train set")
    domain = M.geometry.BB2D(-1.,-1.,1.,1.)

    print(" | Sampling points")
    X_on = np.array([t*A+ (1-t)*B for t in np.random.random(n_train)])
    X_out = M.processing.sampling.sample_bounding_box_2D(domain, n_train)[:,:2]
    print(f" | Generated {X_on.shape[0]} (on), {X_out.shape[0]} (outside)")

    print("Generate test set")
    def distance_to_segment(x):
        XA = x - A
        BA = B - A
        h = min(max(np.dot(XA,BA)/np.dot(BA,BA), 0.), 1.) 
        return np.linalg.norm(XA - h*BA)
    
    m_surf  = min(n_test//4, X_on.shape[0])
    m_other = n_test - m_surf
    X_test = M.processing.sampling.sample_bounding_box_2D(domain, m_other)
    Y_test = np.array([distance_to_segment(x) for x in X_test])
    X_test = np.concatenate((X_test, X_on[np.random.choice(X_on.shape[0], m_surf, replace=False), :]))
    Y_test = np.concatenate((Y_test,np.zeros(m_surf)))
    return X_on, X_out, X_test, Y_test


def generate_cube(n,m):
    raise NotImplementedError

def generate_sphere(n,m):
    raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("which", type=str,
        choices=["square", "circle", "segment", "cube", "sphere"])

    parser.add_argument("-u", "--unsigned", action="store_true")

    parser.add_argument("-n", "--n-train", type=int, default=20000,
        help="number of sample points in inside and outside distributions")

    parser.add_argument("-ni", "--n-surface", type=int, default=10000, 
        help="number of points on the surface. Ignored in no-interior mode")

    parser.add_argument("-nt", "--n-test", type=int, default=1000, \
        help="number of sample points in test set")
    
    parser.add_argument("-visu", help="generates visualization point cloud", action="store_true")

    args = parser.parse_args()

    match args.which:
        case "square":
            if args.unsigned:
                X_on,X_out,X_test,Y_test = generate_square_no_interior(args.n_train, args.n_test)
            else:
                X_in,X_out,X_bd,X_test,Y_test =  generate_square(args.n_train, args.n_surface, args.n_test)
        
        case "circle":
            if args.unsigned:
                X_on,X_out,X_test,Y_test = generate_circle_no_interior(args.n_train, args.n_test)
            else:
                X_in,X_out,X_bd,X_test,Y_test =  generate_circle(args.n_train, args.n_surface, args.n_test)
        
        case "segment":
            args.unsigned = True
            X_on,X_out,X_test,Y_test = generate_segment(args.n_train, args.n_test)

        case "cube":
            X_in,X_out,X_bd,X_test,Y_test =  generate_cube(args.n_train, args.n_surface, args.n_test)
        
        case "sphere":
            X_in,X_out,X_bd,X_test,Y_test =  generate_sphere(args.n_train, args.n_surface, args.n_test)

    if args.visu:
        print("Generate visualization output")
        if args.unsigned:
            pc_train = M.mesh.PointCloud()
            in_out_attr = pc_train.vertices.create_attribute("on", int)
            for i in range(X_on.shape[0]):
                pc_train.vertices.append(geom.Vec(X_on[i,0], X_on[i,1], 0.))
                in_out_attr[i] = -1
            n = len(pc_train.vertices)
            for i in range(X_out.shape[0]):
                pc_train.vertices.append(geom.Vec(X_out[i,0], X_out[i,1], 0.))
                in_out_attr[n+i] = 1
        else:
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

    print("Saving files")
    if args.visu:
        M.mesh.save(pc_train, f"inputs/{args.which}_pts_train.geogram_ascii")
        M.mesh.save(pc_test, f"inputs/{args.which}_pts_test.geogram_ascii")

    if args.unsigned:
        np.save(f"inputs/{args.which}_Xtrain_on.npy", X_on)
    else:
        np.save(f"inputs/{args.which}_Xtrain_in.npy", X_in)
        np.save(f"inputs/{args.which}_Xtrain_bd.npy", X_bd)
    np.save(f"inputs/{args.which}_Xtrain_out.npy", X_out)
    np.save(f"inputs/{args.which}_Xtest.npy", X_test)
    np.save(f"inputs/{args.which}_Ytest.npy", Y_test)