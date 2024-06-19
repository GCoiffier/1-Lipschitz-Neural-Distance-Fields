import mouette as M
from mouette import geometry as geom

import os
import numpy as np
from numpy.random import choice
import argparse
from igl import signed_distance
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

def pseudo_surface_from_polyline(pl):
    nv = len(pl.vertices)
    bary = M.attributes.barycenter(pl)
    bary1 = M.Vec(bary.x, bary.y, 100)
    bary2 = M.Vec(bary.x, bary.y, -100)
    V = list(pl.vertices)+[bary1,bary2]
    F = []
    for a,b in pl.edges:
        if a==0 and b==len(pl.edges)-1:
            ### last edge (n-1, 0) is oriented backwards
            F += [(b,a,nv), (a,b,nv+1)]
        else:
            F += [(a,b,nv), (b,a,nv+1)]
    
    ### Correctly orient normals outward
    ref = M.Vec(-1., 0., 0.) # point outside of the mesh
    volume = 0.
    for (A,B,C) in F:
        pA,pB,pC = V[A], V[B], V[C]
        volume += geom.det_3x3(pB-pA, ref-pA, pC-pA)
    if volume<0:
        F = [(b,a,c) for (a,b,c) in F] # change order of faces
    return np.array(V), np.array(F, dtype=int)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_mesh", type=str, \
        help="path to the input mesh")
    parser.add_argument("-mode", "--mode", default="signed", choices=["signed", "unsigned", "dist", "sal"])
    parser.add_argument("-n", "--n-train", type=int, default=10000)
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
    V,F = pseudo_surface_from_polyline(mesh)
    M.mesh.save(M.mesh.from_arrays(V,F=F), "inputs/pseudo_surface.mesh")

    arrays_to_save = dict()
    mesh_to_save = dict() # if args.visu

    print("Generate train set")
    print(" | Sampling points")
    match args.mode:

        case "unsigned":
            X_on, _ = sample_points_and_normals(mesh, args.n_train)
            X_out = M.sampling.sample_bounding_box_2D(domain, args.n_train)[:,:2]
            print(f" | Generated {X_on.shape[0]} (on), {X_out.shape[0]} (outside)")
            arrays_to_save["Xtrain_on"] = X_on
            arrays_to_save["Xtrain_out"] = X_out
            if args.visu:
                mesh_to_save["pts_on"] = point_cloud_from_array(X_on)
                mesh_to_save["pts_train"] = point_cloud_from_arrays((X_on, -1), (X_out, 1))

        case "signed":
            X_on,_ = sample_points_and_normals(mesh, args.n_train//20)
            mult = 10
            ok = False
            n_ok = 0
            while not ok and n_ok<2:
                X_other = M.sampling.sample_bounding_box_2D(domain, mult*args.n_train)[:,:2]
                Y_other,_,_ = signed_distance(np.pad(X_other, ((0,0), (0,1))), V, F)
                X_in = X_other[Y_other<-1e-3, :][:args.n_train]
                X_out = X_other[Y_other>1e-2, :][:args.n_train]
                ok = (X_in.shape[0] == args.n_train and X_out.shape[0] == args.n_train)
                mult *= 5
                n_ok += 1
            X_in = np.concatenate((X_on,X_in))[:X_out.shape[0],:]
            np.random.shuffle(X_in)
            print(f" | Generated {X_in.shape[0]} (inside), {X_out.shape[0]} (outside)")
            arrays_to_save["Xtrain_in"] = X_in
            arrays_to_save["Xtrain_out"] = X_out
            if args.visu:
                mesh_to_save["pts_train"] = point_cloud_from_arrays((X_out, 1.), (X_in,-1))

        case "dist":
            X_on, N = sample_points_and_normals(mesh, args.n_train//20)
            X_out = M.sampling.sample_bounding_box_2D(domain,args.n_train)[:,:2]
            Y_out,_,_ = signed_distance(np.pad(X_out, ((0,0), (0,1))), V, F)
            X_train = np.concatenate((X_on, X_out))[:args.n_train,:]
            Y_train = np.concatenate((np.zeros(X_on.shape[0], Y_out)))[:args.n_train,:]
            arrays_to_save["Xtrain"] = X_train
            arrays_to_save["Ytrain"] = Y_train
            if args.visu:
                mesh_to_save["pts_train"] = point_cloud_from_array(X_train, Y_train)

        case "sal":
            X_on, Nrml = sample_points_and_normals(mesh, args.n_boundary)
            X, _ = sample_points_and_normals(mesh, args.n_train//2)
            Z1 = X + np.random.normal(0., 1e-2, size = (args.n_train//2, 2))
            Z2 = X + np.random.normal(0., 0.2, size = (args.n_train//2, 2))
            Z = np.concatenate((Z1,Z2))
            Y,_,_ = signed_distance(np.pad(Z, ((0,0), (0,1))), V,F)
            Y = abs(Y)
            arrays_to_save["Xtrain_out"] = Z
            arrays_to_save["Xtrain_on"] = X_on 
            arrays_to_save["Nrml"] = Nrml
            arrays_to_save["Ytrain_out"] = Y
            if args.visu:
                mesh_to_save["X_train"] = point_cloud_from_array(Z,Y)
                mesh_to_save["X_on"] = point_cloud_from_array(X_on)
                mesh_to_save["normals"] = vector_field_from_array(X_on,Nrml,0.1)

    print("Generate test set")
    X_test = M.sampling.sample_bounding_box_2D(domain, args.n_test)
    X_test_on, _ = sample_points_and_normals(mesh, args.n_test_boundary)
    Y_test,_,_ = signed_distance(np.pad(X_test, ((0,0), (0,1))), V,F)
    if args.mode in ["unsigned", "sal"]:
        Y_test = abs(Y_test)
    X_test = np.concatenate((X_test, X_test_on))
    Y_test = np.concatenate((Y_test,np.zeros(args.n_test_boundary)))
    arrays_to_save["Xtest"] = X_test
    arrays_to_save["Ytest"] = Y_test
    if args.visu:
        mesh_to_save["test"] = point_cloud_from_array(X_test, Y_test)

    print("Saving files")
    name = M.utils.get_filename(args.input_mesh)
    if args.visu:
        for mesh_name, mesh in mesh_to_save.items():
            M.mesh.save(mesh, f"inputs/{name}_{mesh_name}.geogram_ascii")
    for ar_name,ar in arrays_to_save.items():
        np.save(f"inputs/{name}_{ar_name}.npy", ar)