import mouette as M
from mouette import geometry as geom

import os
import numpy as np
import argparse
from igl import fast_winding_number_for_meshes
from numba import jit, prange
from common.visualize import point_cloud_from_array

@jit(nopython=True,cache=True)
def point_triangle_distance(P, TRI):
    # function [dist,PP0] = pointTriangleDistance(TRI,P)
    # calculate distance between a point and a triangle in 3D
    # SYNTAX
    #   dist = pointTriangleDistance(TRI,P)
    #   [dist,PP0] = pointTriangleDistance(TRI,P)
    #
    # DESCRIPTION
    #   Calculate the distance of a given point P from a triangle TRI.
    #   Point P is a row vector of the form 1x3. The triangle is a matrix
    #   formed by three rows of points TRI = [P1;P2;P3] each of size 1x3.
    #   dist = pointTriangleDistance(TRI,P) returns the distance of the point P
    #   to the triangle TRI.
    #   [dist,PP0] = pointTriangleDistance(TRI,P) additionally returns the
    #   closest point PP0 to P on the triangle TRI.
    #
    # Author: Gwolyn Fischer
    # Release: 1.0
    # Release date: 09/02/02
    # Release: 1.1 Fixed Bug because of normalization
    # Release: 1.2 Fixed Bug because of typo in region 5 20101013
    # Release: 1.3 Fixed Bug because of typo in region 2 20101014

    # Possible extention could be a version tailored not to return the distance
    # and additionally the closest point, but instead return only the closest
    # point. Could lead to a small speed gain.

    # The algorithm is based on
    # "David Eberly, 'Distance Between Point and Triangle in 3D',
    # Geometric Tools, LLC, (1999)"
    # http:\\www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
    #
    #        ^t
    #  \     |
    #   \reg2|
    #    \   |
    #     \  |
    #      \ |
    #       \|
    #        *P2
    #        |\
    #        | \
    #  reg3  |  \ reg1
    #        |   \
    #        |reg0\
    #        |     \
    #        |      \ P1
    # -------*-------*------->s
    #        |P0      \
    #  reg4  | reg5    \ reg6
    # rewrite triangle in normal form
    B = TRI[0, :]
    E0 = TRI[1, :] - B
    # E0 = E0/sqrt(sum(E0.^2)); %normalize vector
    E1 = TRI[2, :] - B
    # E1 = E1/sqrt(sum(E1.^2)); %normalize vector
    D = B - P
    a = np.dot(E0, E0)
    b = np.dot(E0, E1)
    c = np.dot(E1, E1)
    d = np.dot(E0, D)
    e = np.dot(E1, D)
    f = np.dot(D, D)

    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    # Terible tree of conditionals to determine in which region of the diagram
    # shown above the projection of the point into the triangle-plane lies.
    if (s + t) <= det:
        if s < 0.0:
            if t < 0.0:
                # region4
                if d < 0:
                    t = 0.0
                    if -d >= a:
                        s = 1.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
                else:
                    s = 0.0
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        if -e >= c:
                            t = 1.0
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = -e / c
                            sqrdistance = e * t + f

                            # of region 4
            else:
                # region 3
                s = 0
                if e >= 0:
                    t = 0
                    sqrdistance = f
                else:
                    if -e >= c:
                        t = 1
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 3
        else:
            if t < 0:
                # region 5
                t = 0
                if d >= 0:
                    s = 0
                    sqrdistance = f
                else:
                    if -d >= a:
                        s = 1
                        sqrdistance = a + 2.0 * d + f;  # GF 20101013 fixed typo d*s ->2*d
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
            else:
                # region 0
                invDet = 1.0 / det
                s = s * invDet
                t = t * invDet
                sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
    else:
        if s < 0.0:
            # region 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:  # minimum on edge s+t=1
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                if numer >= denom:
                    s = 1.0
                    t = 0.0
                    sqrdistance = a + 2.0 * d + f;  # GF 20101014 fixed typo 2*b -> 2*d
                else:
                    s = numer / denom
                    t = 1 - s
                    sqrdistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 * e) + f

            else:  # minimum on edge s=0
                s = 0.0
                if tmp1 <= 0.0:
                    t = 1
                    sqrdistance = c + 2.0 * e + f
                else:
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 2
        else:
            if t < 0.0:
                # region6
                tmp0 = b + e
                tmp1 = a + d
                if tmp1 > tmp0:
                    numer = tmp1 - tmp0
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        t = 1.0
                        s = 0
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = numer / denom
                        s = 1 - t
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

                else:
                    t = 0.0
                    if tmp1 <= 0.0:
                        s = 1
                        sqrdistance = a + 2.0 * d + f
                    else:
                        if d >= 0.0:
                            s = 0.0
                            sqrdistance = f
                        else:
                            s = -d / a
                            sqrdistance = d * s + f
            else:
                # region 1
                numer = c + e - b - d
                if numer <= 0:
                    s = 0.0
                    t = 1.0
                    sqrdistance = c + 2.0 * e + f
                else:
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        s = 1.0
                        t = 0.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = numer / denom
                        t = 1 - s
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

    # account for numerical round-off error
    if sqrdistance < 0:
        sqrdistance = 0
    dist = np.sqrt(sqrdistance)
    PP0 = B + s * E0 + t * E1
    return dist, PP0

@jit(nopython=True, parallel=True, cache=True)
def distance_to_surface(P, TRIS):
    d = 1e8
    ntris = TRIS.shape[0]
    for i in prange(ntris):
        d = min(d, point_triangle_distance(P, TRIS[i,:,:])[0])
    return d

@jit(nopython=True,parallel=True, cache=True)
def compute_distances(Q,TRIS):
    nQ = Q.shape[0]
    D = np.zeros(nQ, dtype=float)
    for iQ in prange(nQ):
        D[iQ] = distance_to_surface(Q[iQ,:], TRIS)
    return D

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_mesh", type=str, \
        help="path to the input mesh")

    parser.add_argument("-n", "--n-train", type=int, default=50000, \
        help="number of sample points in train set")

    parser.add_argument("-m", "--n-test", type=int, default=10000, \
        help="number of sample points in test set")
    
    parser.add_argument("-visu", help="generates visualization point cloud", action="store_true")

    args = parser.parse_args()

    os.makedirs("inputs", exist_ok=True)

    print("Load mesh")
    mesh = M.mesh.load(args.input_mesh)
    mesh = M.transform.fit_into_unit_cube(mesh)
    domain = M.geometry.BB3D.of_mesh(mesh, padding=0.3)

    print("Generate train set")
    n_in = args.n_train//2
    n_out = args.n_train//2
    n_surf = n_in//8
    n_other = n_in - n_surf
    print(" | Sample points on surface")
    X_surf = M.processing.sampling.sample_points_from_surface(mesh, n_surf)

    print(" | Sample uniform distribution in domain")
    X_other = M.processing.sampling.sample_bounding_box_3D(domain, 50*n_other)
    print(" | Compute generalized winding number")
    Y_other = fast_winding_number_for_meshes(np.array(mesh.vertices), np.array(mesh.faces, dtype=np.int32), X_other)
    print(f" | WN : [{np.min(Y_other)} ; {np.max(Y_other)}]")

    X_in = X_other[Y_other>0.5][:n_other]
    X_out = X_other[Y_other<=0.5][:n_out]

    print(f"Sampled : {X_in.shape[0] + X_surf.shape[0]} (inside), {X_surf.shape[0]} (surface), {X_out.shape[0]} (outside)")

    print("\nGenerate test set")
    n_surf_test = min(args.n_test//10, n_surf)
    n_other_test = args.n_test - n_surf_test

    print(" | Sampling points on surface")
    X_surf_test = X_surf[np.random.choice(n_surf,n_surf_test, replace=False), :]
    print(" | Sampling uniform distribution in domain")
    X_other_test = M.processing.sampling.sample_bounding_box_3D(domain,n_other_test)

    print(" | Building face array")
    TRIS = np.zeros((len(mesh.faces), 3, 3))
    for i,F in enumerate(mesh.faces):
        pA,pB,pC = (mesh.vertices[x] for x in F)
        TRIS[i,0,:] = pA
        TRIS[i,1,:] = pB
        TRIS[i,2,:] = pC

    print(" | Compute distances")
    D_test = compute_distances(X_other_test,TRIS)
    print(" | Computing generalized winding number and update distance sign")
    Y_other_test = fast_winding_number_for_meshes(np.array(mesh.vertices), np.array(mesh.faces, dtype=np.int32), X_other_test)
    D_test[Y_other_test>0.5] *= -1 # negative distance for points inside the shape

    X_test = np.concatenate((X_surf_test, X_other_test))
    D_test = np.concatenate((np.zeros(X_surf_test.shape[0]), D_test))

    if args.visu:
        print("\nGenerate visualization output")
        pc_in = point_cloud_from_array(X_in)
        pc_out = point_cloud_from_array(X_out)
        pc_surf = point_cloud_from_array(X_surf)
        pc_test = point_cloud_from_array(X_test)
        dist_attr = pc_test.vertices.create_attribute("d", float, dense=True)
        for i in pc_test.id_vertices:
            dist_attr[i] = D_test[i]

    X_in_tot = np.concatenate((X_surf, X_in)) # add surface points to inside
    print("Saving files")
    name = M.utils.get_filename(args.input_mesh)
    if args.visu:
        M.mesh.save(pc_in, f"inputs/{name}_pts_in.xyz")
        M.mesh.save(pc_out, f"inputs/{name}_pts_out.xyz")
        M.mesh.save(pc_surf, f"inputs/{name}_pts_surf.xyz")
        M.mesh.save(pc_test, f"inputs/{name}_pts_test.geogram_ascii")
        M.mesh.save(mesh, f"inputs/{name}_surface.stl")
    np.save(f"inputs/{name}_Xtrain_in.npy", X_in_tot)
    np.save(f"inputs/{name}_Xtrain_out.npy", X_out)
    np.save(f"inputs/{name}_Xtest.npy", X_test)
    np.save(f"inputs/{name}_Ytest.npy", D_test)