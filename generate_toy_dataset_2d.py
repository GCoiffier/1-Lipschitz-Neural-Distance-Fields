import mouette as M
from mouette import geometry as geom

import numpy as np
from numpy.random import choice
import argparse
import cmath

from common.visualize import point_cloud_from_array, point_cloud_from_arrays, vector_field_from_array

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

def generate_square(n_out, n_surf, n_test):
    SIDE = 0.8
    square = M.geometry.BB2D(- SIDE/2, -SIDE/2, SIDE/2, SIDE/2)
    square_mesh = M.mesh.RawMeshData()
    square_mesh.vertices += [
        M.Vec(square.left, square.bottom, 0.),
        M.Vec(square.right, square.bottom, 0.),
        M.Vec(square.right, square.top, 0.),
        M.Vec(square.left, square.top, 0.)
    ]
    square_mesh.edges += [(0,1), (1,2), (2,3), (3,0)]

    def distance_to_square(x):
        d = np.array(abs(x) - np.array([SIDE/2,SIDE/2]))
        return np.linalg.norm(np.maximum(d,0.)) + min(max(d[0], d[1]), 0.)

    print("Generate train set")
    domain = M.geometry.BB2D(-1.,-1.,1.,1.)
    print(" | Sampling points")
    X_on, N = sample_points_and_normals(square_mesh, n_surf)
    X_other = M.processing.sampling.sample_bounding_box_2D(domain, 10*n_out)[:,:2]
    print(" | Discriminate interior from exterior points")
    D = np.array([distance_to_square(x) for x in X_other])
    X_in = X_other[D<0, :][:n_out]
    X_out = X_other[D>0, :]
    X_out = X_out[:X_in.shape[0], :] # same number of points inside and outside
    print(f" | Generated {X_in.shape[0]} (inside), {X_out.shape[0]} (outside), {X_on.shape[0]} (boundary)")

    print("Generate test set")
    n_test_surf  = min(n_test//4, X_on.shape[0])
    n_test_other = n_test - n_test_surf

    X_test = M.processing.sampling.sample_bounding_box_2D(domain, n_test_other)
    Y_test = np.array([distance_to_square(x) for x in X_test])
    X_test = np.concatenate((X_test, X_on[np.random.choice(X_on.shape[0], n_test_surf, replace=False), :]))
    Y_test = np.concatenate((Y_test,np.zeros(n_test_surf)))
    return X_in,X_out,X_on,N,X_test,Y_test


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


def generate_square_distances(n_out, n_surf, n_test):
    SIDE = 0.8
    square = M.geometry.BB2D(- SIDE/2, -SIDE/2, SIDE/2, SIDE/2)
    square_mesh = M.mesh.RawMeshData()
    square_mesh.vertices += [
        M.Vec(square.left, square.bottom, 0.),
        M.Vec(square.right, square.bottom, 0.),
        M.Vec(square.right, square.top, 0.),
        M.Vec(square.left, square.top, 0.)
    ]
    square_mesh.edges += [(0,1), (1,2), (2,3), (3,0)]
    # square_mesh = M.mesh.PolyLine(square_mesh)

    def distance_to_square(x):
        d = np.array(abs(x) - np.array([SIDE/2,SIDE/2]))
        return np.linalg.norm(np.maximum(d,0.)) + min(max(d[0], d[1]), 0.)

    print("Generate train set")
    domain = M.geometry.BB2D(-1.,-1.,1.,1.)
    print(" | Sampling points")
    X_on, _ = sample_points_and_normals(square_mesh, n_surf)
    X_other = M.processing.sampling.sample_bounding_box_2D(domain, n_out)[:,:2]
    print(" | Compute distances")
    Y_train = np.array([distance_to_square(x) for x in X_other])
    X_train = np.concatenate([X_other,X_on])
    Y_train = np.concatenate([Y_train,np.zeros(n_surf)])

    print("Generate test set")
    n_test_surf  = min(n_test//4, X_on.shape[0])
    n_test_other = n_test - n_test_surf

    X_test = M.processing.sampling.sample_bounding_box_2D(domain, n_test_other)
    Y_test = np.array([distance_to_square(x) for x in X_test])
    X_test = np.concatenate((X_test, X_on[np.random.choice(X_on.shape[0], n_test_surf, replace=False), :]))
    Y_test = np.concatenate((Y_test,np.zeros(n_test_surf)))
    return X_train,Y_train, X_test,Y_test
    

def generate_circle(n_out, n_surf, n_test):
    RADIUS = 0.4
    distance_to_circle = lambda x : np.linalg.norm(x) - RADIUS

    print("Generate train set")
    domain = M.geometry.BB2D(-1.,-1.,1.,1.)

    print(" | Sampling points")
    angles = np.random.random(n_surf)*2*np.pi
    X_on = [cmath.rect(RADIUS,ang) for ang in angles]
    X_on = np.array([[x.real, x.imag] for x in X_on]) # convert from complexes to vec2
    N = X_on/np.linalg.norm(X_on,axis=1).reshape((-1,1))
    X_other = M.processing.sampling.sample_bounding_box_2D(domain, 10*n_out)[:,:2]

    print(" | Discriminate interior from exterior points")    
    D = np.array([distance_to_circle(x) for x in X_other])
    X_in = X_other[D<0, :][:n_out]
    X_out = X_other[D>0, :]
    X_out = X_out[:X_in.shape[0], :] # same number of points inside and outside
    print(f" | Generated {X_in.shape[0]} (inside), {X_out.shape[0]} (outside), {X_on.shape[0]} (boundary)")

    print("Generate test set")
    n_test_surf  = n_test//4
    n_test_other = n_test - n_test_surf

    X_test = M.processing.sampling.sample_bounding_box_2D(domain, n_test_other)
    Y_test = np.array([distance_to_circle(x) for x in X_test])

    angles = np.random.random(n_test_surf)*2*np.pi
    X_test_bd = [cmath.rect(RADIUS,ang) for ang in angles]
    X_test_bd = np.array([[x.real, x.imag] for x in X_test_bd]) # convert from complexes to vec2

    X_test = np.concatenate((X_test, X_test_bd))
    Y_test = np.concatenate((Y_test,np.zeros(n_test_surf)))
    return X_in,X_out,X_on,N,X_test,Y_test


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


def generate_circle_distances(n_out, n_surf, n_test):
    RADIUS = 0.4
    distance_to_circle = lambda x : np.linalg.norm(x) - RADIUS

    print("Generate train set")
    domain = M.geometry.BB2D(-1.,-1.,1.,1.)

    print(" | Sampling points")
    angles = np.random.random(n_surf)*2*np.pi
    X_on = [cmath.rect(RADIUS,ang) for ang in angles]
    X_on = np.array([[x.real, x.imag] for x in X_on]) # convert from complexes to vec2
    X_other = M.processing.sampling.sample_bounding_box_2D(domain, 10*n_out)[:,:2]

    print(" | Compute distances")    
    Y_train = np.array([distance_to_circle(x) for x in X_other])
    X_train = np.concatenate((X_other, X_on))
    Y_train = np.concatenate((Y_train, np.zeros(n_surf)))

    print("Generate test set")
    n_test_surf  = n_test//4
    n_test_other = n_test - n_test_surf

    X_test = M.processing.sampling.sample_bounding_box_2D(domain, n_test_other)
    Y_test = np.array([distance_to_circle(x) for x in X_test])

    angles = np.random.random(n_test_surf)*2*np.pi
    X_test_bd = [cmath.rect(RADIUS,ang) for ang in angles]
    X_test_bd = np.array([[x.real, x.imag] for x in X_test_bd]) # convert from complexes to vec2

    X_test = np.concatenate((X_test, X_test_bd))
    Y_test = np.concatenate((Y_test,np.zeros(n_test_surf)))
    return X_train, Y_train, X_test,Y_test


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

def generate_segment_distances(n_train, n_on, n_test):
    A = np.array([-0.5,-0.2])
    B = np.array([0.5,0.2])
    def distance_to_segment(x):
        XA = x - A
        BA = B - A
        h = min(max(np.dot(XA,BA)/np.dot(BA,BA), 0.), 1.) 
        return np.linalg.norm(XA - h*BA)

    print("Generate train set")
    domain = M.geometry.BB2D(-1.,-1.,1.,1.)

    print(" | Sampling points")
    X_on = np.array([t*A+ (1-t)*B for t in np.random.random(n_on)])
    X_out = M.processing.sampling.sample_bounding_box_2D(domain, n_train)[:,:2]
    X_train = np.concatenate((X_on, X_out))
    Y_train = np.array([distance_to_segment(x) for x in X_train])
    
    print("Generate test set")
    m_surf  = min(n_test//4, X_on.shape[0])
    m_other = n_test - m_surf
    X_test = M.processing.sampling.sample_bounding_box_2D(domain, m_other)
    Y_test = np.array([distance_to_segment(x) for x in X_test])
    X_test = np.concatenate((X_test, X_on[np.random.choice(X_on.shape[0], m_surf, replace=False), :]))
    Y_test = np.concatenate((Y_test,np.zeros(m_surf)))
    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("which", type=str,
        choices=["square", "circle", "segment"])

    parser.add_argument("-mode", "--mode", choices=["signed", "unsigned", "dist"], default="signed")

    parser.add_argument("-no", "--n-train", type=int, default=20000,
        help="number of sample points in inside and outside distributions")

    parser.add_argument("-ni", "--n-surface", type=int, default=10000, 
        help="number of points on the surface. Ignored in no-interior mode")

    parser.add_argument("-nt", "--n-test", type=int, default=1000, \
        help="number of sample points in test set")
    
    parser.add_argument("-visu", help="generates visualization point cloud", action="store_true")

    args = parser.parse_args()
    

    match args.mode:
        case "unsigned":
            X_on,X_out,X_test,Y_test = {
                "square" : generate_square_no_interior,
                "circle" : generate_circle_no_interior,
                "segment" : generate_segment
            }[args.which](args.n_train, args.n_test)
            arrays_to_save = {
                f"inputs/{args.which}_Xtrain_on.npy" : X_on,
                f"inputs/{args.which}_Xtrain_out.npy" : X_out,
                f"inputs/{args.which}_Xtest.npy" : X_test,
                f"inputs/{args.which}_Ytest.npy" : Y_test,
            }
            if args.visu:
                pc_to_save = {
                    f"inputs/{args.which}_pctrain.geogram_ascii" : point_cloud_from_arrays((X_on, -1), (X_out,1)),
                    f"inputs/{args.which}_pctest.geogram_ascii" : point_cloud_from_array(X_test, Y_test)
                }
        
        case "signed":
            X_in,X_out,X_on,N,X_test,Y_test = {
                "square" : generate_square,
                "circle" : generate_circle
            }[args.which](args.n_train, args.n_surface, args.n_test)
            arrays_to_save = {
                f"inputs/{args.which}_Xtrain_in.npy" : X_in,
                f"inputs/{args.which}_Xtrain_on.npy" : X_on,
                f"inputs/{args.which}_Nrml.npy" : N,
                f"inputs/{args.which}_Xtrain_out.npy" : X_out,
                f"inputs/{args.which}_Xtest.npy" : X_test,
                f"inputs/{args.which}_Ytest.npy" : Y_test,
            }
            if args.visu:
                pc_to_save = {
                    f"inputs/{args.which}_pctrain.geogram_ascii" : point_cloud_from_arrays((X_in, -1), (X_out,1), (X_on,0.)),
                    f"inputs/{args.which}_normals.mesh" : vector_field_from_array(X_on, N, 0.1) ,
                    f"inputs/{args.which}_pctest.geogram_ascii" : point_cloud_from_array(X_test, Y_test)
                }

        case "dist":
            X_train, Y_train, X_test, Y_test = {
                "square" : generate_square_distances,
                "circle" : generate_circle_distances,
                "segment" : generate_segment_distances
            }[args.which](args.n_train, args.n_surface, args.n_test)
            arrays_to_save = {
                f"inputs/{args.which}_Xtrain.npy" : X_train,
                f"inputs/{args.which}_Ytrain.npy" : Y_train,
                f"inputs/{args.which}_Xtest.npy" : X_test,
                f"inputs/{args.which}_Ytest.npy" : Y_test,
            }
            if args.visu:
                pc_to_save = {
                    f"inputs/{args.which}_pctrain.geogram_ascii" : point_cloud_from_array(X_train, Y_train),
                    f"inputs/{args.which}_pctest.geogram_ascii" : point_cloud_from_array(X_test, Y_test)
                }

    print("Saving files")
    for path,ar in arrays_to_save.items():
        np.save(path, ar)
    if args.visu:
        for path,m in pc_to_save.items():
            M.mesh.save(m, path)
