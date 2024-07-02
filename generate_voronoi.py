import mouette as M

import os
import numpy as np
import argparse

from common.visualize import point_cloud_from_array, point_cloud_from_arrays, vector_field_from_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-no", "--n-train", type=int, default=10_000)
    parser.add_argument("-visu", help="generates visualization point cloud", action="store_true")
    args = parser.parse_args()

    os.makedirs("inputs", exist_ok=True)

    print("Generate train set")
    domain = M.geometry.BB2D(-1,-1,1,1)
    X_train = M.sampling.sample_bounding_box_2D(domain, args.n_train)
    X_train_on = np.repeat(M.sampling.sample_bounding_box_2D(domain,10), args.n_train//10, axis=0)

    arrays_to_save = {
        "voronoi_Xtrain_on" : X_train_on,
        "voronoi_Xtrain_out" : X_train 
    }
    
    if args.visu:
        M.mesh.save(point_cloud_from_arrays((X_train,-1), (X_train_on,1)), f"inputs/points.geogram_ascii")

    print("Saving files")
    for file,ar in arrays_to_save.items():
        np.save(f"inputs/{file}.npy", ar)
