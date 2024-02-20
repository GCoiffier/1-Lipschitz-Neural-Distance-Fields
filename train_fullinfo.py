import os
import argparse
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import mouette as M

from common.models import *
from common.callback import *
from common.training import Trainer
from common.utils import get_device, get_BB

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # dataset parameters
    parser.add_argument("dataset", type=str)
    parser.add_argument("-o", "--output-name", type=str, default="")
    parser.add_argument("--unsigned", type=str, default="")

    # model parameters
    parser.add_argument("-model", "--model", choices=["mlp", "siren", "ortho", "sll"], default="mlp")
    parser.add_argument("-n-layers", "--n-layers", type=int, default=8)
    parser.add_argument("-n-hidden", "--n-hidden", type=int, default=32)

    # optimization parameters
    parser.add_argument("-ne", "--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument('-bs',"--batch-size", type=int, default=200, help="Batch size")
    parser.add_argument("-tbs", "--test-batch-size", type = int, default = 5000, help="Batch size on test set")
    parser.add_argument("-weik", "--eikonal-weight", type = float, default=0., help="weight for eikonal loss")
    parser.add_argument("-wtv", "--total-variation-weight", type = float, default=0., help="weight for total variation loss")

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
        eikonal_weight = args.eikonal_weight,
        tv_weight = args.total_variation_weight,
        optimizer = "adam",
        learning_rate = 5e-4,
        output_folder = os.path.join("output", args.output_name if len(args.output_name)>0 else args.dataset)
    )
    os.makedirs(config.output_folder, exist_ok=True)
    print("DEVICE:", config.device)

    #### Load dataset ####
    X_train = np.load(os.path.join("inputs", f"{args.dataset}_Xtrain.npy"))
    X_train = torch.Tensor(X_train).to(config.device)
    Y_train = np.load(os.path.join("inputs", f"{args.dataset}_Ytrain.npy")).reshape((X_train.shape[0], 1))
    Y_train = torch.Tensor(Y_train).to(config.device)
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=config.batch_size, shuffle=True)
    print(f"Succesfully loaded train set: {X_train.shape}\n")

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

    DIM = X_train.shape[1] # dimension of the dataset (2 or 3)

    #### Create model and setup trainer
    model = select_model(args.model, DIM, args.n_layers, args.n_hidden, final_activ=torch.nn.Tanh).to(config.device)
    print("MODEL", model)
    print("PARAMETERS:", count_parameters(model))

    trainer = Trainer(train_loader, test_loader, config)
    callbacks = []
    callbacks.append(LoggerCB(os.path.join(config.output_folder, "log.csv")))
    if config.checkpoint_freq>0:
        callbacks.append(CheckpointCB([x for x in range(0, config.n_epochs, config.checkpoint_freq) if x>0]))
        if DIM==2:
            plot_domain = get_BB(X_train, DIM, pad=0.5)
            callbacks.append(Render2DCB(config.output_folder, config.checkpoint_freq, plot_domain, res=1000))

    trainer.add_callbacks(*callbacks)
    trainer.train_full_info(model)

    #### Save final model
    path = os.path.join(config.output_folder, "model_final.pt")
    save_model(model, path)
