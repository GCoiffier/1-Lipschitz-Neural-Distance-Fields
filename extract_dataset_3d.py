import mouette as M

import os
import numpy as np
import argparse
from igl import fast_winding_number_for_meshes, signed_distance
from common.visualize import point_cloud_from_array

def extract_train_point_cloud(n_bd, n_train, mesh, domain):
    print(" | Sample points on surface")
    X_bd, N_bd = M.processing.sampling.sample_points_from_surface(mesh, n_bd, return_normals=True)
    print(" | Sample uniform distribution in domain")
    domain.pad(0.05,0.05,0.05)
    X_other1 = M.processing.sampling.sample_bounding_box_3D(domain, 50*n_train)
    domain.pad(0.95,0.95,0.95)
    X_other2 = M.processing.sampling.sample_bounding_box_3D(domain, 30*n_train)
    X_other = np.concatenate((X_other1, X_other2))
    np.random.shuffle(X_other)
    print(" | Compute generalized winding number")
    Y_other = fast_winding_number_for_meshes(np.array(mesh.vertices), np.array(mesh.faces, dtype=np.int32), X_other)
    print(f" | WN : [{np.min(Y_other)} ; {np.max(Y_other)}]")
    X_in = X_other[Y_other>0.5][:n_train]
    X_out = X_other[Y_other<0.5][:n_train]
    print(f"Sampled : {X_in.shape[0]} (inside), {X_out.shape[0]} (outside), {X_bd.shape[0]} (surface)")
    return X_bd, N_bd, X_in, X_out

def extract_train_point_cloud_unsigned(n_pt, mesh, domain):
    print(" | Sample points on surface")
    X_on = M.processing.sampling.sample_points_from_surface(mesh, n_pt)
    print(" | Sample uniform distribution in domain")
    domain.pad(0.05,0.05,0.05)
    X_out1 = M.processing.sampling.sample_bounding_box_3D(domain, n_pt//2)
    domain.pad(0.95,0.95,0.95)
    X_out2 = M.processing.sampling.sample_bounding_box_3D(domain, n_pt//2)
    X_out = np.concatenate((X_out1, X_out2))
    print(f"Sampled : {X_on.shape[0]} (surface), {X_out.shape[0]} (outside)")
    return X_on, X_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_mesh", type=str, \
        help="path to the input mesh")
    parser.add_argument("--unsigned", action="store_true")
    parser.add_argument("-no", "--n-train", type=int, default=100_000)
    parser.add_argument("-ni", "--n-boundary", type=int, default=10_000)
    parser.add_argument("-nt", "--n-test",  type=int, default=10_000)
    parser.add_argument("-nti", "--n-test-boundary", type=int, default=2000)
    parser.add_argument("-visu", help="generates visualization point cloud", action="store_true")
    args = parser.parse_args()

    os.makedirs("inputs", exist_ok=True)

    print("Load mesh")
    mesh = M.mesh.load(args.input_mesh)
    mesh = M.transform.fit_into_unit_cube(mesh)
    mesh = M.transform.translate(mesh, -np.mean(mesh.vertices, axis=0))
    domain = M.geometry.BB3D.of_mesh(mesh)

    print("Generate train set")
    if args.unsigned:
        X_on, X_out = extract_train_point_cloud_unsigned(args.n_train, mesh, domain)
    else:
        X_bd, N_bd, X_in, X_out = extract_train_point_cloud(args.n_boundary, args.n_train, mesh, domain)

    print("\nGenerate test set")
    print(" | Sampling points on surface")
    X_surf_test = M.processing.sampling.sample_points_from_surface(mesh, args.n_test_boundary)
    print(" | Sampling uniform distribution in domain")
    X_other_test1 = M.processing.sampling.sample_bounding_box_3D(M.geometry.BB3D.of_mesh(mesh,padding=0.1), args.n_test//2)
    X_other_test2 = M.processing.sampling.sample_bounding_box_3D(M.geometry.BB3D.of_mesh(mesh,padding=1.), args.n_test//2)
    X_other_test = np.concatenate((X_other_test1, X_other_test2))
    
    print(" | Building face array")
    TRIS = np.zeros((len(mesh.faces), 3, 3))
    for i,F in enumerate(mesh.faces):
        pA,pB,pC = (mesh.vertices[x] for x in F)
        TRIS[i,0,:] = pA
        TRIS[i,1,:] = pB
        TRIS[i,2,:] = pC

    print(" | Compute distances")
    D_test,_,_ = signed_distance(X_other_test, np.array(mesh.vertices), np.array(mesh.faces, dtype=np.int32))
    if args.unsigned:
        D_test = abs(D_test)
    X_test = np.concatenate((X_surf_test, X_other_test))
    D_test = np.concatenate((np.zeros(X_surf_test.shape[0]), D_test))

    if args.visu:
        print("\nGenerate visualization output")
        if args.unsigned:
            pc_on = point_cloud_from_array(X_on)
        else:
            pc_in = point_cloud_from_array(X_in)
            pc_surf = point_cloud_from_array(X_bd)
            nrml_poly = M.mesh.PolyLine()
            for i in range(X_bd.shape[0]):
                p1 = X_bd[i,:]
                p2 = p1 + N_bd[i,:]/20
                nrml_poly.vertices += [p1, p2]
                nrml_poly.edges.append((2*i, 2*i+1))

        pc_out = point_cloud_from_array(X_out)
        pc_test = point_cloud_from_array(X_test)
        dist_attr = pc_test.vertices.create_attribute("d", float, dense=True)
        for i in pc_test.id_vertices:
            dist_attr[i] = D_test[i]

    print("Saving files")
    name = M.utils.get_filename(args.input_mesh)
    if args.visu:
        if args.unsigned:
            M.mesh.save(pc_on, f"inputs/{name}_pts_on.xyz")
        else:
            M.mesh.save(pc_in, f"inputs/{name}_pts_in.xyz")
            M.mesh.save(pc_surf, f"inputs/{name}_pts_surf.xyz")
            M.mesh.save(nrml_poly, f"inputs/{name}_normals.mesh")
        M.mesh.save(pc_out, f"inputs/{name}_pts_out.xyz")
        M.mesh.save(pc_test, f"inputs/{name}_pts_test.geogram_ascii")
        M.mesh.save(mesh, f"inputs/{name}_surface.stl")
    if args.unsigned:
        np.save(f"inputs/{name}_Xtrain_on.npy", X_on)
    else:
        np.save(f"inputs/{name}_Xtrain_in.npy", X_in)
        np.save(f"inputs/{name}_Xtrain_bd.npy", X_bd)
        np.save(f"inputs/{name}_Normals_bd.npy", N_bd)
    np.save(f"inputs/{name}_Xtrain_out.npy", X_out)
    np.save(f"inputs/{name}_Xtest.npy", X_test)
    np.save(f"inputs/{name}_Ytest.npy", D_test)