import os
from types import SimpleNamespace
import argparse
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

import mouette as M

from common.models import *
from common.visualize import point_cloud_from_arrays
from common.training import Trainer
from common.utils import get_device, get_BB
from common.callback import *

"""
SAL: Sign Agnostic Learning of Shapes from Raw Data
"""

if __name__ == "__main__":

    #### Commandline ####
    parser = argparse.ArgumentParser(
        prog="SALD trainer",
        description="Reimplementation of the Signed-Agnostic distance function learning of Atzmon and Lipman"
    )

    # dataset parameters
    parser.add_argument("dataset", type=str, help="Name of the dataset to consider. Should have been generated using the 'sal' mode of the dataset extraction scripts.")
    parser.add_argument("-o", "--output-name", type=str, default="")

    # model parameters
    parser.add_argument("-model", "--model", choices=["mlp", "siren", "ortho", "sll"], default="mlp", help="Network architecture to consider. 1-Lipschitz architectures are also available")
    parser.add_argument("-n-layers", "--n-layers", type=int, default=8, help="number of layers in the network")
    parser.add_argument("-n-hidden", "--n-hidden", type=int, default=32, help="size of each layers in the network")

    # optimization parameters
    parser.add_argument("-ne", "--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument('-bs',"--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("-tbs", "--test-batch-size", type = int, default = 5000, help="Batch size on test set")
    parser.add_argument("--metric", choices=["l0", "l2"], default="l2", help="norm to consider for the SAL learning")
    
    # misc
    parser.add_argument("-cp", "--checkpoint-freq", type=int, default=10)
    parser.add_argument("-cpu", action="store_true")
    args = parser.parse_args()

    #### Config ####
    config = SimpleNamespace(
        device = get_device(args.cpu),
        n_epochs = args.epochs,
        checkpoint_freq = args.checkpoint_freq,
        batch_size = args.batch_size,
        test_batch_size = args.test_batch_size,
        metric = args.metric,
        attach_weight = args.attach_weight,
        grad_weight = args.grad_weight,
        optimizer = "adam",
        learning_rate = 5e-4,
        output_folder = os.path.join("output", args.output_name if len(args.output_name)>0 else args.dataset)
    )
    os.makedirs(config.output_folder, exist_ok=True)
    print("DEVICE:", config.device)

    #### Load train set ####
    X_train_out = np.load(os.path.join("inputs", f"{args.dataset}_Xtrain_out.npy"))
    X_train_out = torch.Tensor(X_train_out).to(config.device)
    Y_train_out = np.load(os.path.join("inputs", f"{args.dataset}_Ytrain_out.npy"))
    Y_train_out = torch.Tensor(Y_train_out).to(config.device)
    train_loader_out = DataLoader(TensorDataset(X_train_out,Y_train_out), batch_size=config.batch_size, shuffle=True)
    
    X_train_on = np.load(os.path.join("inputs", f"{args.dataset}_Xtrain_on.npy"))
    X_train_on = torch.Tensor(X_train_on).to(config.device)
    Normals = np.load(os.path.join("inputs", f"{args.dataset}_Nrml.npy"))
    Normals = torch.Tensor(Normals).to(config.device)
    ratio = X_train_out.shape[0]//X_train_on.shape[0]
    train_loader_on = DataLoader(TensorDataset(X_train_on,Normals), batch_size=config.batch_size//ratio, shuffle=True)

    print(f"Succesfully loaded train set:\n", 
          f" Outside: {X_train_out.shape}\n",
          f" Surface: {X_train_on.shape}")

    #### Load test set ####
    X_test,Y_test = None,None
    test_loader = None
    X_test_path = os.path.join("inputs", f"{args.dataset}_Xtest.npy")
    Y_test_path = os.path.join("inputs",f"{args.dataset}_Ytest.npy")
    if os.path.exists(X_test_path) and os.path.exists(Y_test_path):
        X_test = np.load(os.path.join("inputs", f"{args.dataset}_Xtest.npy"))
        Y_test = np.load(os.path.join("inputs", f"{args.dataset}_Ytest.npy")).reshape((X_test.shape[0], 1))
        X_test = torch.Tensor(X_test).to(config.device)
        Y_test = torch.Tensor(Y_test).to(config.device)
        test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=config.test_batch_size)
        print(f"Succesfully loaded test set: {X_test.shape}\n")

    DIM : int = X_train_out.shape[1] # dimension of the dataset (2 or 3)

    #### Create model
    model = select_model(
        args.model, DIM, args.n_layers, args.n_hidden
        ).to(config.device)
    print("PARAMETERS:", count_parameters(model))

    M.mesh.save(
        point_cloud_from_arrays(
            (X_train_out.detach().cpu(),1), 
            (X_train_on.detach().cpu(),-1)), 
        os.path.join(config.output_folder, f"pc_0.geogram_ascii")
    )

    #### Building and calling the trainer ####
    callbacks = []
    callbacks.append(LoggerCB(os.path.join(config.output_folder, "log.csv")))
    callbacks.append(CheckpointCB([x for x in range(0, config.n_epochs, config.checkpoint_freq) if x>0]))
    plot_domain = get_BB(X_train_on, DIM, pad=0.2)
    if DIM==2:
        callbacks.append(Render2DCB(config.output_folder, config.checkpoint_freq, plot_domain, output_gradient_norm=False, res=800))
    else:
        callbacks.append(MarchingCubeCB(config.output_folder, config.checkpoint_freq, plot_domain))
    trainer = Trainer((train_loader_out, train_loader_on), test_loader, config)
    trainer.add_callbacks(*callbacks)
    trainer.train_sal(model)

    #### Save final model
    path = os.path.join(config.output_folder, "model_final.pt")
    save_model(model, path)