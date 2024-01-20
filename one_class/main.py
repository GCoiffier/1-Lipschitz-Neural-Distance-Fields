import os
import argparse
from types import SimpleNamespace

import mouette as M

from common.model import DenseLipNetwork, save_model
from common.dataset import PointCloudDataset
from common.visualize import *
from common.training import train
from common.utils import get_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", type=str)
    parser.add_argument("-n", "--n-iter", type=int, default=10, help="Number of iterations")
    parser.add_argument('-bs',"--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("-ne", "--epochs", type=int, default=10, help="Number of epochs per iteration")
    parser.add_argument("-cpu", action="store_true")
    args = parser.parse_args()

    #### Config ####
    config = SimpleNamespace(
        dim = 3,
        device = get_device(args.cpu),
        n_iter = args.n_iter,
        batch_size = args.batch_size,
        test_batch_size = 1000,
        epochs = args.epochs,
        loss_margin = 0.01, # m
        loss_regul = 1000., # lambda
        optimizer = "adam",
        learning_rate = 0.005,
        NR_maxiter = 3,
        output_folder = os.path.join("output", args.dataset)
    )
    os.makedirs(config.output_folder, exist_ok=True)
    print("DEVICE:", config.device)

    #### Load dataset ####
    dataset = PointCloudDataset(args.dataset, config)
    plot_domain = dataset.object_BB
    plot_domain.pad(0.5, 0.5, 0.5)

    #### Create model and setup trainer
    # archi = [(3,256), (256,256), (256,256), (256,256), (256,1)]
    # archi = [(3,64), (64,64), (64,64), (64,64), (64,1)]
    archi = [(3,32), (32,32), (32,32), (32,32), (32,1)]

    model = DenseLipNetwork(
        archi, group_sort_size=0,
    ).to(config.device)

    for n in range(config.n_iter):
        print("ITERATION", n+1)
        pc = point_cloud_from_tensor(dataset.X_train_in.detach().cpu(), dataset.X_train_out.detach().cpu())
        M.mesh.save(pc, os.path.join(config.output_folder, f"pc_{n}.geogram_ascii"))

        train(model, dataset, config)
        singular_values = parameter_singular_values(model)
        print("Singular values:")
        for sv in singular_values:
            print(sv)
        print()
        model_path = os.path.join(config.output_folder, f"model_{n+1}.pt")
        save_model(model, archi, model_path)
        dataset.update_complementary_distribution(model, config.NR_maxiter)