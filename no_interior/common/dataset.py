import os
import numpy as np
import mouette as M

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F

class PointCloudDataset(M.Logger):

    def __init__(self, name, config):
        super().__init__("Dataset", verbose=True)
        self.paths = {
            "Xtrain_on" : os.path.join("inputs", f"{name}_Xtrain_on.npy"),
            "Xtrain_out" : os.path.join("inputs", f"{name}_Xtrain_out.npy"),
            "Xtest" : os.path.join("inputs", f"{name}_Xtest.npy"),
            "Ytest" : os.path.join("inputs", f"{name}_Ytest.npy")
        }
        self.config = config

        self._object_bb = None
        self._domain = None

        # Load data
        self.log("Loading dataset...")
        self.X_train_on = np.load(self.paths["Xtrain_on"])
        self.X_train_on = torch.Tensor(self.X_train_on).to(self.config.device)
        self._train_loader_on = None

        self.X_train_out = np.load(self.paths["Xtrain_out"])
        self.X_train_out = torch.Tensor(self.X_train_out).to(self.config.device)
        self._train_loader_out = None

        self.X_test = np.load(self.paths["Xtest"])
        self.Y_test = np.load(self.paths["Ytest"]).reshape((self.X_test.shape[0], 1))
        self.X_test = torch.Tensor(self.X_test).to(self.config.device)
        self.Y_test = torch.Tensor(self.Y_test).to(self.config.device)
        test_data = TensorDataset(self.X_test, self.Y_test)
        self.test_loader = DataLoader(test_data, batch_size=self.config.test_batch_size)

        self.log(f"Succesfully loaded:\n", 
                f"On: {self.X_train_on.shape}\n", 
                f"Outside: {self.X_train_out.shape}\n", 
                f"Test: {self.X_test.shape}")

    @property
    def object_BB(self):
        if self._object_bb is None :
            if self.X_train_on is None : return None
            vmin = torch.min(self.X_train_on, dim=0)[0].cpu()
            vmax = torch.max(self.X_train_on, dim=0)[0].cpu() 
            if self.config.dim == 2:
                self._object_bb = M.geometry.BB2D(float(vmin[0]), float(vmin[1]), float(vmax[0]), float(vmax[1]))
            elif self.config.dim == 3:
                self._object_bb = M.geometry.BB3D(*vmin, *vmax)
        return self._object_bb

    @property
    def domain(self):
        if self._domain is None :
            if self.X_train_on is None : return None
            vmin = torch.min(self.X_train_out, dim=0)[0].cpu()
            vmax = torch.max(self.X_train_out, dim=0)[0].cpu()
            if self.config.dim==2: 
                self._domain = M.geometry.BB2D(float(vmin[0]), float(vmin[1]), float(vmax[0]), float(vmax[1])) 
            elif self.config.dim==3:
                self._domain = M.geometry.BB3D(*vmin, *vmax)
        return self._domain

    @property
    def train_loader(self):
        if self._train_loader_on is None or self._train_loader_out is None:
            self._train_loader_on = DataLoader(TensorDataset(self.X_train_on), batch_size=self.config.batch_size, shuffle=True)
            self._train_loader_out = DataLoader(TensorDataset(self.X_train_out), batch_size=self.config.batch_size, shuffle=True)
        return zip(self._train_loader_on, self._train_loader_out)

    @property
    def train_size(self):
        return self.X_train_on.shape[0] // self.config.batch_size
    
    def __len__(self):
        return self.train_size

    def clip_to_domain(self, X):
        if self.config.dim==2:
            X[:,0] = torch.clip(X[:,0], self.domain.left, self.domain.right)
            X[:,1] = torch.clip(X[:,1], self.domain.bottom, self.domain.top)
        elif self.config.dim==3:
            for i in range(3):
                X[:,i] = torch.clip(X[:,i], self.domain.min_coords[i], self.domain.max_coords[i])
        return X
    
    def update_complementary_distribution(self, model, maxiter:int=3, level_set:float=-1e-3):
        """
        Perform Newton-Raphson iteration on the points to make them closer

        Args:
            model: pytorch model representing the sdf
        """
        step_size = 1. / maxiter
        Xt = self.X_train_out.clone()
        learning_rate = torch.rand((Xt.shape[0],1)).to(self.config.device)

        for _ in range(maxiter):

            Xt = Xt.detach()
            Xt.requires_grad = True

            # Compute signed distances
            y = model(Xt)
            ysum = torch.mean(y)
            ysum.backward()
            
            # retrieve gradient of the function
            grad = Xt.grad
            grad_norm_squared = torch.sum(grad**2, axis=1).reshape((Xt.shape[0],1))
            
            grad = grad / (grad_norm_squared + 1e-8)
            target = y + level_set
            # target = F.relu(target)
            Xt = Xt - step_size * learning_rate * target * grad
            
            # clipping to domain
            Xt = self.clip_to_domain(Xt)
        self.X_train_out = Xt.detach()

        self._train_loader_out = None # reset loader

