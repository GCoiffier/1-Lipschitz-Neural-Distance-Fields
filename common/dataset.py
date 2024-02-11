import os
import numpy as np
import mouette as M

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F

from .utils import get_BB

class PointCloudDataset(M.Logger):

    def __init__(self, name, config):
        super().__init__("Dataset", verbose=True)
        self.paths = {
            "Xtrain_in" : os.path.join("inputs", f"{name}_Xtrain_in.npy"),
            "Xtrain_out" : os.path.join("inputs", f"{name}_Xtrain_out.npy"),
            "Xtrain_bd" : os.path.join("inputs", f"{name}_Xtrain_bd.npy"),
            "Normals_bd" : os.path.join("inputs", f"{name}_Normals_bd.npy"),
            "Xtest" : os.path.join("inputs", f"{name}_Xtest.npy"),
            "Ytest" : os.path.join("inputs", f"{name}_Ytest.npy")
        }
        self.config = config

        self._object_bb = None
        self._domain = None

        # Load data
        self.log("Loading dataset...")
        self.X_train_in = np.load(self.paths["Xtrain_in"])
        self.X_train_in = torch.Tensor(self.X_train_in).to(self.config.device)
        self._train_loader_in = None

        self.X_train_out = np.load(self.paths["Xtrain_out"])
        self.X_train_out = torch.Tensor(self.X_train_out).to(self.config.device)
        self._train_loader_out = None

        self.X_train_bd = np.load(self.paths["Xtrain_bd"])
        self.X_train_bd = torch.Tensor(self.X_train_bd).to(self.config.device)
        self._train_loader_bd = None

        found_normals = os.path.exists(self.paths["Normals_bd"])
        if found_normals:
            self.Normals_bd = np.load(self.paths["Normals_bd"])
            self.Normals_bd = torch.Tensor(self.Normals_bd).to(self.config.device)
            self._train_loader_normals = None
        elif self.config.normal_weight>0. :
            self.log("No normals found. Removing normal reconstruction loss")
            self.config.normal_weight = 0.
    
        self.X_test = np.load(self.paths["Xtest"])
        self.Y_test = np.load(self.paths["Ytest"]).reshape((self.X_test.shape[0], 1))
        self.X_test = torch.Tensor(self.X_test).to(self.config.device)
        self.Y_test = torch.Tensor(self.Y_test).to(self.config.device)
        test_data = TensorDataset(self.X_test, self.Y_test)
        self.test_loader = DataLoader(test_data, batch_size=self.config.test_batch_size)

        self.log(f"Succesfully loaded:\n", 
                f"Inside: {self.X_train_in.shape}\n", 
                f"Outside: {self.X_train_out.shape}\n",
                f"Boundary: {self.X_train_bd.shape}\n",
                f"Test: {self.X_test.shape}")
        if not self.config.attach_weight>0.:
            # merge X_in and X_bd
            n = self.X_train_in.shape[0] - self.X_train_bd.shape[0]
            self.X_train_in = torch.concatenate((self.X_train_in[:n], self.X_train_bd))
            assert self.X_train_in.shape[0] == self.X_train_out.shape[0]

    @property
    def object_BB(self):
        if self._object_bb is None :
            self._object_bb = get_BB(self.X_train_bd, self.config.dim)
        return self._object_bb

    @property
    def domain(self):
        if self._domain is None :
            self._domain = get_BB(self.X_train_out, self.config.dim)
        return self._domain

    @property
    def train_loader(self):
        if self.config.normal_weight>0.:
            if any((self._train_loader_in is None,
                    self._train_loader_out is None, 
                    self._train_loader_bd is None)):
                self._train_loader_in = DataLoader(TensorDataset(self.X_train_in), batch_size=self.config.batch_size, shuffle=True)
                self._train_loader_out = DataLoader(TensorDataset(self.X_train_out), batch_size=self.config.batch_size, shuffle=True)
                ratio = self.X_train_in.shape[0]//self.X_train_bd.shape[0]
                self._train_loader_bd = DataLoader(TensorDataset(self.X_train_bd), batch_size=self.config.batch_size//ratio)
                self._train_loader_normals = DataLoader(TensorDataset(self.X_train_bd), batch_size=self.config.batch_size//ratio)
            return zip(self._train_loader_in, self._train_loader_out, self._train_loader_bd, self._train_loader_normals)
        
        elif self.config.attach_weight>0.:
            if any((self._train_loader_in is None,
                    self._train_loader_out is None, 
                    self._train_loader_bd is None)):
                self._train_loader_in = DataLoader(TensorDataset(self.X_train_in), batch_size=self.config.batch_size, shuffle=True)
                self._train_loader_out = DataLoader(TensorDataset(self.X_train_out), batch_size=self.config.batch_size, shuffle=True)
                ratio = self.X_train_in.shape[0]//self.X_train_bd.shape[0]
                self._train_loader_bd = DataLoader(TensorDataset(self.X_train_bd), batch_size=self.config.batch_size//ratio)
            return zip(self._train_loader_in, self._train_loader_out, self._train_loader_bd)
        
        else:
            if self._train_loader_in is None or self._train_loader_out is None:
                self._train_loader_in = DataLoader(TensorDataset(self.X_train_in), batch_size=self.config.batch_size, shuffle=True)
                self._train_loader_out = DataLoader(TensorDataset(self.X_train_out), batch_size=self.config.batch_size, shuffle=True)
            return zip(self._train_loader_in, self._train_loader_out)

    @property
    def train_size(self):
        return self.X_train_in.shape[0] // self.config.batch_size
    
    def __len__(self):
        return self.train_size


class PointCloudDataset_NoInterior(M.Logger):

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
            self._object_bb = get_BB(self.X_train_on, self.config.dim)
        return self._object_bb

    @property
    def domain(self):
        if self._domain is None :
            self._domain = get_BB(self.X_train_out, self.config.dim)
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
    

class PointCloudDataset_FullInfo(M.Logger):

    def __init__(self, name, config):
        super().__init__("Dataset", verbose=True)
        self.paths = {
            "Xtrain" : os.path.join("inputs", f"{name}_Xtrain.npy"),
            "Ytrain" : os.path.join("inputs", f"{name}_Ytrain.npy"),
            "Xtest" : os.path.join("inputs", f"{name}_Xtest.npy"),
            "Ytest" : os.path.join("inputs", f"{name}_Ytest.npy")
        }
        self.config = config

        self._object_bb = None
        self._domain = None

        # Load data
        self.log("Loading dataset...")
        self.X_train = np.load(self.paths["Xtrain"])
        self.X_train = torch.Tensor(self.X_train).to(self.config.device)
        self.Y_train = np.load(self.paths["Ytrain"]).reshape((self.X_train.shape[0], 1))
        self.Y_train = torch.Tensor(self.Y_train).to(self.config.device)
        train_data = TensorDataset(self.X_train, self.Y_train)
        self._train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)

        self.X_test = np.load(self.paths["Xtest"])
        self.Y_test = np.load(self.paths["Ytest"]).reshape((self.X_test.shape[0], 1))
        self.X_test = torch.Tensor(self.X_test).to(self.config.device)
        self.Y_test = torch.Tensor(self.Y_test).to(self.config.device)
        test_data = TensorDataset(self.X_test, self.Y_test)
        self.test_loader = DataLoader(test_data, batch_size=self.config.test_batch_size)

        self.log(f"Succesfully loaded:\n", 
                f"Train: {self.X_train.shape}\n",  
                f"Test: {self.X_test.shape}")

    @property
    def object_BB(self):
        if self._object_bb is None :
            X_in = self.X_train[np.squeeze(self.Y_train)<=0, :]
            self._object_bb = get_BB(X_in, self.config.dim)
        return self._object_bb

    @property
    def domain(self):
        if self._domain is None :
            self._domain = get_BB(self.X_train, self.config.dim)
        return self._domain

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def train_size(self):
        return self.X_train.shape[0] // self.config.batch_size
    
    def __len__(self):
        return self.train_size