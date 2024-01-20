import os
import numpy as np
import mouette as M
import cmath

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F

from abc import abstractproperty, abstractmethod

class _BasePointCloudDataset(M.Logger):
    def __init__(self, name, config):
        super().__init__("Dataset", verbose=True)
        self.paths = {
            "Xtrain_in" : os.path.join("inputs", f"{name}_Xtrain_in.npy"),
            "Xtrain_out" : os.path.join("inputs", f"{name}_Xtrain_out.npy"),
            "Xtest" : os.path.join("inputs", f"{name}_Xtest.npy"),
            "Ytest" : os.path.join("inputs", f"{name}_Ytest.npy")
        }
        self.config = config

        self._object_bb = None
        self._domain = None
        
        self.X_train_in = None
        self.X_train_out = None
        self.train_loader_in, self.train_loader_out = None, None
        
        self.X_test = None
        self.Y_test = None
        self.test_loader = None
        self.load_dataset()

    @abstractproperty
    def object_BB(self):
        raise NotImplementedError

    @abstractproperty
    def domain(self):
        raise NotImplementedError

    @property
    def train_size(self):
        return self.X_train_in.shape[0]

    @property
    def train_loader(self):
        return enumerate(zip(self.train_loader_in, self.train_loader_out))

    def load_dataset(self):
        self.log("Loading Dataset...")
        ### Load test dataset
        self.X_test = np.load(self.paths["Xtest"])
        self.Y_test = np.load(self.paths["Ytest"]).reshape((self.X_test.shape[0], 1))

        self.X_test  = torch.Tensor(self.X_test).to(self.config.device)
        self.Y_test  = torch.Tensor(self.Y_test).to(self.config.device)

        ### Load and create train dataset
        self.X_train_in = np.load(self.paths["Xtrain_in"])
        self.X_train_in = torch.Tensor(self.X_train_in).to(self.config.device)
        self.train_loader_in = DataLoader(TensorDataset(self.X_train_in), batch_size=self.config.batch_size, shuffle=True)

        self.X_train_out = np.load(self.paths["Xtrain_out"])
        self.X_train_out = torch.Tensor(self.X_train_out).to(self.config.device)
        
        self.train_loader_out = DataLoader(TensorDataset(self.X_train_out), batch_size=self.config.batch_size, shuffle=True)

        test_data = TensorDataset(self.X_test, self.Y_test)
        self.test_loader = DataLoader(test_data, batch_size=self.config.test_batch_size)
        self.log("...Dataset loading complete")

    @abstractmethod
    def update_complementary_distribution(self, model, maxiter, level_set=-1e-3):
        raise NotImplementedError

class PointCloudDataset2D(_BasePointCloudDataset):

    def __init__(self, name, config):
        super().__init__(name, config)

    @property
    def object_BB(self) -> M.geometry.BB2D:
        if self._object_bb is None :
            if self.X_train_in is None : return None
            vmin = torch.min(self.X_train_in, dim=0)[0]
            vmax = torch.max(self.X_train_in, dim=0)[0] 
            return M.geometry.BB2D(float(vmin[0]), float(vmin[1]), float(vmax[0]), float(vmax[1]))
        return self._object_bb
    
    @property
    def domain(self) -> M.geometry.BB2D:
        if self._domain is None :
            if self.X_train_out is None : return None
            vmin = torch.min(self.X_train_out, dim=0)[0]
            vmax = torch.max(self.X_train_out, dim=0)[0] 
            self._domain = M.geometry.BB2D(float(vmin[0]), float(vmin[1]), float(vmax[0]), float(vmax[1]))
        return self._domain

    def update_complementary_distribution(self, model, maxiter, level_set=-1e-3):
        """
        Perform Newton-Raphson iteration on the points to make them closer

        Args:
            sdf: pytorch model
        """
        step_size = 1. / maxiter
        Xt = self.X_train_out.clone()
        learning_rate = torch.rand((Xt.shape[0],1)).to(self.config.device)

        for _ in range(maxiter):

            Xt = Xt.detach()
            Xt.requires_grad = True

            # Compute signed distances
            y = torch.mean(model(Xt))
            y.backward()
            
            # retrieve gradient of the function
            grad = Xt.grad
            grad_norm_squared = torch.sum(grad**2, axis=1).reshape((Xt.shape[0],1))            
            grad = grad / (grad_norm_squared + 1e-8)
            target = F.relu(-y + level_set)
            Xt = Xt + step_size * learning_rate * target * grad
            
            # clipping to domain
            Xt[:,0] = torch.clip(Xt[:,0], self.domain.left, self.domain.right)
            Xt[:,1] = torch.clip(Xt[:,1], self.domain.bottom, self.domain.top)

        self.X_train_out = Xt.detach()
        self.train_loader_out = DataLoader(TensorDataset(self.X_train_out), batch_size=self.config.batch_size, shuffle=True)


class PointCloudDataset(_BasePointCloudDataset):

    def __init__(self, name, config):
        super().__init__(name, config)

    @property
    def object_BB(self):
        if self._object_bb is None :
            if self.X_train_in is None : return None
            vmin = torch.min(self.X_train_in, dim=0)[0]
            vmax = torch.max(self.X_train_in, dim=0)[0]
            self._object_bb = M.geometry.BB3D(*vmin, *vmax)
        return self._object_bb
    
    @property
    def domain(self):
        if self._domain is None :
            if self.X_train_in is None : return None
            vmin = torch.min(self.X_train_in, dim=0)[0]
            vmax = torch.max(self.X_train_in, dim=0)[0]
            self._domain = M.geometry.BB3D(*vmin, *vmax)
        return self._domain

    def update_complementary_distribution(self, model, maxiter, level_set=-1e-3):
        """
        Perform Newton-Raphson iteration on the points to make them closer

        Args:
            sdf: pytorch model
        """
        step_size = 1. / maxiter
        Xt = self.X_train_out.clone()
        learning_rate = torch.rand((Xt.shape[0],1))

        for _ in range(maxiter):

            Xt = Xt.detach()
            Xt.requires_grad = True

            # Compute signed distances
            y = model(Xt)
            Gy = torch.ones_like(y)
            y.backward(Gy,retain_graph=True)
            
            # retrieve gradient of the function
            grad = Xt.grad
            grad_norm_squared = torch.sum(grad**2, axis=1).reshape((Xt.shape[0],1))            
            grad = grad / (grad_norm_squared + 1e-8)
            target = F.relu(-y + level_set)
            Xt = Xt + step_size * learning_rate * target * grad
            
            # clipping to domain
            for i in range(3):
                Xt[:,i] = torch.clip(Xt[:,i], self.domain.min_coords[i], self.domain.max_coords[i])

        self.X_train_out = Xt.detach()
        self.train_loader_out = DataLoader(TensorDataset(self.X_train_out), batch_size=self.config.batch_size, shuffle=True)