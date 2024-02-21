import os
from types import SimpleNamespace
import argparse
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import autograd

import mouette as M

from common.models import *
from common.visualize import point_cloud_from_arrays
from common.training import Trainer
from common.utils import *
from common.callback import *

def adversarial_resampling(model, n_pts, domain, device):
    X_out = M.sampling.sample_bounding_box_2D(domain,50*n_pts)
    Y_out, grad = forward_in_batches(model, X_out, device, compute_grad=True)
    W = np.linalg.norm(grad, axis=1)
    W = np.exp(-5*W)
    W /= np.sum(W)
    chosen = np.random.choice(X_out.shape[0],size=10*n_pts,replace=False,p=W)
    X_out = X_out[chosen,:]
    Y_out = model(torch.Tensor(X_out)).detach().cpu().numpy().squeeze()
    X_in = X_out[Y_out<=0, :]
    X_out = X_out[Y_out>0, :]
    n_samples = min((X_in.shape[0], X_out.shape[0], n_pts))
    print(n_pts, n_samples)
    return X_in[:n_samples,:], X_out[:n_samples,:]

if __name__ == "__main__":

    #### Commandline ####
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument("dataset", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("-o", "--output-name", type=str, default="")

    # sampling parameters
    parser.add_argument("-no", "--no", type=int, default=10_000)
  
    # optimization parameters
    parser.add_argument("-ne", "--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument('-bs',"--batch-size", type=int, default=200, help="Batch size")
    parser.add_argument("-tbs", "--test-batch-size", type = int, default = 5000, help="Batch size on test set")
    parser.add_argument("-lr", "--learning-rate", type=float, default=5e-4, help="Adam's learning rate")
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
        learning_rate = args.learning_rate,
        output_folder = os.path.join("output", args.output_name if len(args.output_name)>0 else args.dataset)
    )
    os.makedirs(config.output_folder, exist_ok=True)
    print("DEVICE:", config.device)

    DIM : int = None

    model = load_model(args.model, config.device)

    #### Load train set ####
    X_train_on = np.load(os.path.join("inputs", f"{args.dataset}_Xtrain_on.npy"))
    X_train_on = torch.Tensor(X_train_on).to(config.device)
    DIM = X_train_on.shape[1]

    X_train_out, X_train_in = adversarial_resampling(model, args.no, get_BB(X_train_on, DIM, pad=0.5), config.device)
    X_train_out = torch.Tensor(X_train_out).to(config.device)
    X_train_in = torch.Tensor(X_train_in).to(config.device)
    loader_in = DataLoader(TensorDataset(X_train_in), batch_size=config.batch_size, shuffle=True)    
    loader_out = DataLoader(TensorDataset(X_train_out), batch_size=config.batch_size, shuffle=True)
    
    ratio = X_train_in.shape[0]//X_train_on.shape[0]
    Normals = None
    if config.normal_weight>0. and os.path.exists(normal_path := os.path.join("inputs", f"{args.dataset}_Nrml.npy")):
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
    print("PARAMETERS:", count_parameters(model))

    #### Export point cloud for visualization ####
    pc = point_cloud_from_arrays(
        (X_train_in.detach().cpu(), -1.),
        (X_train_out.detach().cpu(), 1.),
        (X_train_on.detach().cpu(), 0.)
    )
    M.mesh.save(pc, os.path.join(config.output_folder, f"pc_adv.geogram_ascii"))

    #### Building and calling the trainer ####
    callbacks = []
    callbacks.append(LoggerCB(os.path.join(config.output_folder, "log.csv")))
    if config.checkpoint_freq>0:
        callbacks.append(CheckpointCB(range(0, config.n_epochs, config.checkpoint_freq)))
        if DIM==2:
            plot_domain = get_BB(X_train_on, DIM, pad=0.5)
            callbacks.append(Render2DCB(config.output_folder, config.checkpoint_freq, plot_domain, res=1000))
    
    trainer = Trainer((loader_in, loader_out, loader_bd), test_loader, config)
    trainer.add_callbacks(*callbacks)
    trainer.train_lip(model)

    #### Save final model
    path = os.path.join(config.output_folder, "model_intermediate.pt")
    save_model(model, path)