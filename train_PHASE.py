import os
import time
from types import SimpleNamespace
import argparse
import numpy as np
import mouette as M
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import TensorDataset, DataLoader

from common.models import *
from common.visualize import point_cloud_from_array
from common.utils import get_device, get_BB
from common.callback import *

"""
_Phase Transitions, Distance Functions, and Implicit Neural Representations_, Yaron Lipman
"""

def doubleWellPotential(s):
    return (s ** 2) - 2 * (s.abs()) + 1.

class PhaseTrainer(M.Logger):

    def __init__(self, points, normals, test_loader, config):
        super().__init__("Training")
        self.config = config
        self.X = points
        self.N = normals
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        self.scheduler = None # torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)
        self.callbacks = []
        self.metrics = dict()
    
    def add_callbacks(self, *args):
        for cb in args:
            assert(isinstance(cb, Callback))
            self.callbacks.append(cb)

    def evaluate_model(self, model):
        """Evaluates the model on the test dataset.
        Computes the mean square error between actual distances and output of the model
        """
        if self.test_loader is None: return
        test_loss = 0.
        testlossfun = nn.MSELoss() # mean square error
        for inputs,labels in self.test_loader:
            outputs = model(inputs)
            loss = testlossfun(outputs, labels)
            test_loss += loss.item()
        self.metrics["test_loss"] = test_loss
        for cb in self.callbacks:
            cb.callOnEndTest(self, model)

    def train(self, model):
        for epoch in range(self.config.n_epochs):  # loop over the dataset multiple times
            self.metrics["epoch"] = epoch+1
            for cb in self.callbacks:
                cb.callOnBeginTrain(self, model)
            t0 = time.time()
            train_loss = dict()
            total_loss = 0.
            
            selected_pts = np.random.choice(self.X.shape[0], self.config.batch_size, replace=False)
            ball_centers = self.X[selected_pts].numpy()
            ball_points = np.random.normal(ball_centers, 1e-3, size=(self.config.batch_size, self.config.dim))
            X_on = torch.Tensor(ball_points).to(self.config.device)
            if self.N is not None and self.config.normal_weight>0:
                N_on = self.N[selected_pts].to(self.config.device)
            X_out = 2*torch.rand_like(X_on)-1.

            self.optimizer.zero_grad() # zero the parameter gradients
            # forward + backward + optimize
            X_on.requires_grad = True
            X_out.requires_grad = True

            ### Reconstruction loss
            y_on = model(X_on)
            fit_loss = y_on.abs().mean()
            train_loss["fit"] = self.config.attach_weight * fit_loss.item()
            total_loss += self.config.attach_weight * fit_loss

            ### WCH loss
            y_out = model(X_out) # forward computation
            batch_grad = torch.autograd.grad(y_out, X_out, grad_outputs=torch.ones_like(y_out), create_graph=True)[0]
            W_u = doubleWellPotential(y_out)
            WCH_loss = (self.config.epsilon * batch_grad.norm(2, dim=-1)**2 + W_u).mean()
            train_loss["WCH"] = WCH_loss.item()
            total_loss += WCH_loss

             ### Normal fitting loss
            if self.N is not None and self.config.normal_weight>0.:
                grad = autograd.grad(outputs=y_on, inputs=X_on,
                        grad_outputs=torch.ones_like(y_on).to(self.config.device),
                        create_graph=True, retain_graph=True)[0]
                loss_normals =  (N_on - grad).norm(1,dim=-1).mean()
                train_loss["normals"] = self.config.normal_weight*loss_normals.item()
                total_loss += self.config.normal_weight*loss_normals.item()
            total_loss.backward() # call back propagation
            self.optimizer.step()
            for cb in self.callbacks:
                cb.callOnEndForward(self, model)

            self.metrics["train_loss"] = train_loss
            self.metrics["epoch_time"] = time.time() - t0
            for cb in self.callbacks:
                cb.callOnEndTrain(self, model)
            self.evaluate_model(model)

if __name__ == "__main__":

    #### Commandline ####
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument("dataset", type=str)
    parser.add_argument("-o", "--output-name", type=str, default="")

    # model parameters
    parser.add_argument("-model", "--model", choices=["phase", "mlp", "sll"], default="phase")
    parser.add_argument("-n-layers", "--n-layers", type=int, default=8)
    parser.add_argument("-n-hidden", "--n-hidden", type=int, default=512)
    parser.add_argument("-FF", action="store_true")

    # optimization parameters
    parser.add_argument("-ne", "--epochs", type=int, default=2000, help="Number of epochs")
    parser.add_argument('-bs',"--batch-size", type=int, default=1000, help="Batch size")
    parser.add_argument("-tbs", "--test-batch-size", type = int, default = 5000, help="Batch size on test set")
    parser.add_argument("-wa", "--attach-weight", type=float, default=0.3, help="weight for fitting loss. Lambda from paper")
    parser.add_argument("-weps", "--wch-epsilon", type=float, default=0.01, help="epsilon from paper")
    parser.add_argument("-wn", "--normal-weight", type=float, default=0., help="weigt for normal reconstruction loss. Mu from paper")
    # misc
    parser.add_argument("-cp", "--checkpoint-freq", type=int, default=500)
    parser.add_argument("-cpu", action="store_true")
    args = parser.parse_args()

    #### Config ####
    config = SimpleNamespace(
        dim = 0,
        device = get_device(args.cpu),
        n_epochs = args.epochs,
        checkpoint_freq = args.checkpoint_freq,
        batch_size = args.batch_size,
        test_batch_size = args.test_batch_size,
        attach_weight = args.attach_weight,
        epsilon = args.wch_epsilon,
        normal_weight = args.normal_weight,
        optimizer = "adam",
        learning_rate = 5e-4,
        output_folder = os.path.join("output", args.output_name if len(args.output_name)>0 else args.dataset)
    )
    os.makedirs(config.output_folder, exist_ok=True)
    print("DEVICE:", config.device)

    #### Load train set ####
    X_train_on = np.load(os.path.join("inputs", f"{args.dataset}_Xtrain_on.npy"))
    X_train_on = torch.Tensor(X_train_on)
    Normals = None
    if config.normal_weight>0. and os.path.exists(normal_path := os.path.join("inputs", f"{args.dataset}_Nrml.npy")):
        Normals = np.load(os.path.join("inputs", f"{args.dataset}_Nrml.npy"))
        Normals = torch.Tensor(Normals)
    elif config.normal_weight>0. :
        print("No normals found. Removing normal reconstruction loss")
        config.normal_weight = 0.
   
    print(f"Succesfully loaded train set:\n", 
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

    config.dim = X_train_on.shape[1] # dimension of the dataset (2 or 3)

    #### Create model
    model = select_model(
        args.model, config.dim, args.n_layers, args.n_hidden, FF=args.FF
        ).to(config.device)
    print(model)
    print("PARAMETERS:", count_parameters(model))

    M.mesh.save(point_cloud_from_array(X_train_on.cpu().detach().numpy()), "pc_0.geogram_ascii")

    #### Building and calling the trainer ####
    callbacks = []
    callbacks.append(LoggerCB(os.path.join(config.output_folder, "log.csv")))
    callbacks.append(CheckpointCB([x for x in range(0, config.n_epochs, config.checkpoint_freq) if x>0]))
    plot_domain = get_BB(X_train_on, config.dim, pad=0.2)
    if config.dim==2:
        callbacks.append(Render2DCB(config.output_folder, config.checkpoint_freq, plot_domain, output_gradient_norm=True, res=800))
    else:
        callbacks.append(MarchingCubeCB(config.output_folder, config.checkpoint_freq, plot_domain))
    trainer = PhaseTrainer(X_train_on, Normals, test_loader, config)
    trainer.add_callbacks(*callbacks)
    trainer.train(model)

    #### Save final model
    path = os.path.join(config.output_folder, "model_final.pt")
    save_model(model, path)