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

if __name__ == "__main__":

    #### Commandline ####
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument("dataset", type=str)
    parser.add_argument("-o", "--output-name", type=str, default="")
    parser.add_argument("--unsigned", action="store_true")

    # model parameters
    parser.add_argument("-model","--model", choices=["ortho", "sll"], default="ortho")
    parser.add_argument("-n-layers", "--n-layers", type=int, default=8)
    parser.add_argument("-n-hidden", "--n-hidden", type=int, default=32)

    # optimization parameters
    parser.add_argument("-ne", "--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument('-bs',"--batch-size", type=int, default=200, help="Batch size")
    parser.add_argument("-tbs", "--test-batch-size", type = int, default = 5000, help="Batch size on test set")
    parser.add_argument("-lm", "--loss-margin", type=float, default=1e-2)
    parser.add_argument("-lmbd", "--loss-lambda", type=float, default=100.)
    
    parser.add_argument("-wa", "--attach-weight", type=float, default=0., help="weight for fitting loss. Has no effect if --unsigned")
    parser.add_argument("-wn", "--normal-weight", type=float, default=0., help="weigt for normal reconstruction loss. Has no effect if --unsigned")
    parser.add_argument("-weik", "--eikonal-weight", type = float, default=0., help="weight for eikonal loss")
    parser.add_argument("-wgnorm", "--gnorm-weight", type = float, default=0., help="weight for max gradient norm loss")
    
    # misc
    parser.add_argument("-cp", "--checkpoint-freq", type=int, default=10)
    parser.add_argument("-cpu", action="store_true")
    args = parser.parse_args()

    #### Config ####
    config = SimpleNamespace(
        signed = not args.unsigned,
        device = get_device(args.cpu),
        n_epochs = args.epochs,
        checkpoint_freq = args.checkpoint_freq,
        batch_size = args.batch_size,
        test_batch_size = args.test_batch_size,
        loss_margin = args.loss_margin,
        loss_regul = args.loss_lambda,
        attach_weight = args.attach_weight,
        normal_weight = args.normal_weight,
        eikonal_weight = args.eikonal_weight,
        gnorm_weight = args.gnorm_weight,
        optimizer = "adam",
        learning_rate = 1e-4,
        output_folder = os.path.join("output", args.output_name if len(args.output_name)>0 else args.dataset)
    )
    os.makedirs(config.output_folder, exist_ok=True)
    print("DEVICE:", config.device)

    DIM : int = None

    #### Load train set ####
    if config.signed: # SIGNED MODE
        X_train_in = np.load(os.path.join("inputs", f"{args.dataset}_Xtrain_in.npy"))
        X_train_in = torch.Tensor(X_train_in).to(config.device)
        loader_in = DataLoader(TensorDataset(X_train_in), batch_size=config.batch_size, shuffle=True)
        
        X_train_out = np.load(os.path.join("inputs", f"{args.dataset}_Xtrain_out.npy"))
        X_train_out = torch.Tensor(X_train_out).to(config.device)
        loader_out = DataLoader(TensorDataset(X_train_out), batch_size=config.batch_size, shuffle=True)

        X_train_on = np.load(os.path.join("inputs", f"{args.dataset}_Xtrain_on.npy"))
        X_train_on = torch.Tensor(X_train_on).to(config.device)
        ratio = X_train_in.shape[0]//X_train_on.shape[0]
        
        Normals = None
        if config.normal_weight>0. and os.path.exists(normal_path := os.path.join("inputs", f"{args.dataset}_Normals_bd.npy")):
            Normals = np.load(normal_path)
            Normals = torch.Tensor(Normals).to(config.device)
        elif config.normal_weight>0. :
            print("No normals found. Removing normal reconstruction loss")
            config.normal_weight = 0.
        
        if Normals is None or (not config.normal_weight>0.):
            loader_bd = DataLoader(TensorDataset(X_train_on), batch_size=config.batch_size//ratio, shuffle=True)
        else:
            loader_bd = DataLoader(TensorDataset(X_train_on, Normals), batch_size=config.batch_size//ratio, shuffle=True)

        print(f"Succesfully loaded train set:\n", 
              f"Inside: {X_train_in.shape}\n", 
              f"Outside: {X_train_out.shape}\n",
              f"Surface: {X_train_on.shape}")
        if Normals is not None:
            print(f"Normals: {Normals.shape}")
        
    else: # UNSIGNED MODE
        # Attach and normal losses are not supported in unsigned mode
        config.attach_weight = 0.
        config.normal_weight = 0.
        X_train_on = np.load(os.path.join("inputs", f"{args.dataset}_Xtrain_on.npy"))
        X_train_on = torch.Tensor(X_train_on).to(config.device)
        loader_on = DataLoader(TensorDataset(X_train_on), batch_size=config.batch_size, shuffle=True)
        X_train_out = np.load(os.path.join("inputs", f"{args.dataset}_Xtrain_out.npy"))
        X_train_out = torch.Tensor(X_train_out).to(config.device)
        loader_out = DataLoader(TensorDataset(X_train_out), batch_size=config.batch_size, shuffle=True)
        train_loader = zip(loader_on, loader_out)
        
        print(f"Succesfully loaded train set:\n", 
              f"Outside: {X_train_out.shape}\n",
              f"Surface: {X_train_on.shape}")
        

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

    DIM = X_train_out.shape[1] # dimension of the dataset (2 or 3)

    #### Create model ####
    if args.model.lower() == "ortho":
        model = DenseLipNetwork(
            DIM, args.n_hidden, args.n_layers,
            group_sort_size=0, niter_spectral=3, niter_bjorck=15
        ).to(config.device)

    elif args.model.lower() == "sll":
        model = DenseSDP(DIM, args.n_hidden, args.n_layers).to(config.device)
    print("PARAMETERS:", count_parameters(model))


    #### Export point cloud for visualization ####
    if config.signed:
        pc = point_cloud_from_arrays(
            (X_train_in.detach().cpu(), -1.),
            (X_train_out.detach().cpu(), 1.),
            (X_train_on.detach().cpu(), 0.)
        )
    else:
        pc = point_cloud_from_arrays(
            (X_train_on.detach().cpu(), -1.),
            (X_train_out.detach().cpu(), 1.)
        )
    M.mesh.save(pc, os.path.join(config.output_folder, f"pc_0.geogram_ascii"))

    #### Building and calling the trainer ####
    callbacks = []
    callbacks.append(LoggerCB(os.path.join(config.output_folder, "log.txt")))
    callbacks.append(CheckpointCB([x for x in range(0, config.n_epochs, config.checkpoint_freq) if x>0]))
    if DIM==2:
        plot_domain = get_BB(X_train_on, DIM, pad=0.5) if config.signed else get_BB(X_train_on, DIM, pad=0.5)
        callbacks.append(Render2DCB(config.output_folder, config.checkpoint_freq, plot_domain, res=1000))
    # callbacks.append(ComputeSingularValuesCB(config.checkpoint_freq))
    callbacks.append(UpdateHkrRegulCB({1 : 1., 10 : 10., 20: 100.}))
    
    if config.signed:
        trainer = Trainer((loader_in, loader_out, loader_bd), test_loader, config)
        trainer.add_callbacks(*callbacks)
        trainer.train_lip(model)
    else:
        trainer = Trainer((loader_on, loader_out), test_loader, config)
        trainer.add_callbacks(*callbacks)
        trainer.train_lip_unsigned(model)

    #### Save final model
    path = os.path.join(config.output_folder, "model_final.pt")
    save_model(model, path)
