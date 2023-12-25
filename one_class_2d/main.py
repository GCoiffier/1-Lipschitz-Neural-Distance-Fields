import os
import argparse
import numpy as np
from types import SimpleNamespace

import mouette as M

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import DenseLipNetwork, count_parameters
from src.dataset import PointCloudDataset
from src.visualize import point_cloud_from_tensor, render_sdf, render_gradient_norm
from src.training import train

def get_device(force_cpu):
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    else:
        return torch.device("cuda")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", type=str)
    parser.add_argument("-n", "--n-iter", type=int, default=20, help="Number of iterations")
    parser.add_argument('-bs',"--batch-size", type=int, default=300, help="Batch size")
    parser.add_argument("-ne", "--epochs", type=int, default=20, help="Number of epochs per iteration")
    parser.add_argument("-cpu", action="store_true")
    args = parser.parse_args()

    #### Config ####
    config = SimpleNamespace(
        device = get_device(args.cpu),
        n_iter = args.n_iter,
        batch_size = args.batch_size,
        test_batch_size = 1000,
        epochs = args.epochs,
        loss_margin = 1e-2, # m
        loss_regul = 100., # lambda
        optimizer = "adam",
        learning_rate = 1e-3,
        NR_maxiter = 3,
        output_folder = os.path.join("output", args.dataset)
    )
    os.makedirs(config.output_folder, exist_ok=True)
    print("DEVICE:", config.device)

    #### Load dataset ####
    dataset = PointCloudDataset(args.dataset, config)
    plot_domain = dataset.object_BB()
    plot_domain.pad(0.5, 0.8)

    #### Create model and setup trainer
    
    # model = DenseLipNetwork(
    #     [(2,256), (256,256), (256,256), (256,256), (256,1)], 
    #     group_sort_size=2, niter_spectral=3, niter_bjorck=20
    # ).to(config.device)

    # model = DenseLipNetwork(
    #     [(2,64), (64,64), (64,64), (64,64), (64,1)], 
    #     group_sort_size=0, niter_spectral=3, niter_bjorck=15
    # ).to(config.device)

    model = DenseLipNetwork(
        [(2,32), (32,32), (32,32), (32,32), (32,1)], 
        group_sort_size=0, niter_spectral=10, niter_bjorck=300
    ).to(config.device)

    print("PARAMETERS:", count_parameters(model))

    for n in range(config.n_iter):
        print("ITERATION", n+1)
        pc = point_cloud_from_tensor(dataset.X_train_in.detach().cpu(), dataset.X_train_out.detach().cpu())
        M.mesh.save(pc, os.path.join(config.output_folder, f"pc_{n}.geogram_ascii"))

        train(model, dataset, config)

        render_path = os.path.join(config.output_folder, f"render_{n}.png")
        render_sdf(render_path, model, plot_domain, config.device)
        grad_path = os.path.join(config.output_folder, f"grad_{n}.png")
        render_gradient_norm(grad_path, model, plot_domain, config.device)
        dataset.update_complementary_distribution(model, config.NR_maxiter)

        if n==0:
            config.loss_margin = 1e-3