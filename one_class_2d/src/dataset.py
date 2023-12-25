import os
import numpy as np
import mouette as M
import cmath

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F

from tqdm import trange

class PointCloudDataset(M.Logger):
    def __init__(self, name, config):
        super().__init__("Dataset", verbose=True)
        self.paths = {
            "train" : os.path.join("inputs", f"{name}_Xtrain.npy"),
            "Xtest" : os.path.join("inputs", f"{name}_Xtest.npy"),
            "Ytest" : os.path.join("inputs", f"{name}_Ytest.npy")
        }
        self.config = config
        self.domain : M.geometry.BB2D = None
        
        self.X_train_in = None
        self.X_train_out = None
        self.train_loader_in, self.train_loader_out = None, None

        self.X_test, self.Y_test = None, None
        
        self.test_loader = None
        self.load_dataset()

    def object_BB(self):
        if self.X_train_in is None : return None
        vmin = torch.min(self.X_train_in, dim=0)[0]
        vmax = torch.max(self.X_train_in, dim=0)[0]
        return M.geometry.BB2D(float(vmin[0]), float(vmin[1]), float(vmax[0]), float(vmax[1]))

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
        self.X_train_in = np.load(self.paths["train"])
        self.X_train_in = torch.Tensor(self.X_train_in).to(self.config.device)

        self.log(f"Found {self.X_train_in.shape[0]} training examples in dataset")
     
        objBB = self.object_BB()
        pad = 0.5
        self.domain = M.geometry.BB2D( objBB.left -pad, objBB.bottom - pad, objBB.right + pad, objBB.top + pad)

        self.log(f"Generating training examples out of distribution...")
        X_out = self.generate_complementary_distribution_uniform(self.X_train_in.shape[0])
        # X1 = self.generate_complementary_distribution_uniform(self.X_train_in.shape[0]//2)
        # X2 = self.generate_complementary_distribution_shell(self.X_train_in.shape[0]//2)
        # X_out = np.concatenate((X1,X2), axis=0)

        self.X_train_out = torch.Tensor(X_out).to(self.config.device)
        
        self.log(f"...Generated {self.X_train_out.shape[0]} training examples out of distribution")

        self.train_loader_in = DataLoader(TensorDataset(self.X_train_in), batch_size=self.config.batch_size, shuffle=True)
        self.train_loader_out = DataLoader(TensorDataset(self.X_train_out), batch_size=self.config.batch_size, shuffle=True)

        test_data = TensorDataset(self.X_test, self.Y_test)
        self.test_loader = DataLoader(test_data, batch_size=self.config.test_batch_size)
        self.log("...Dataset loading complete")

    def generate_complementary_distribution_uniform(self, n):
        n2 = int(1.5*n)
        pts = M.processing.sampling.sample_bounding_box_2D(self.domain, n2)
        keep = np.ones(n2, dtype=bool)
        for i in trange(n2):
            pt = pts[i,:]
            keep[i] = (torch.min(torch.norm(self.X_train_in - pt, dim=1)) >= self.config.loss_margin)
        pts = pts[keep][:n,:]
        return pts

    def generate_complementary_distribution_shell(self,n):
        X = np.zeros((n,2))
        for i in trange(n):
            center = self.X_train_in[i,:]
            good = False
            while not good:
                angle = 2*np.pi*np.random.random()
                v = cmath.rect(3*self.config.loss_margin, angle)
                sample = torch.Tensor([center[0] + v.real, center[1] + v.imag])
                if torch.min(torch.norm(self.X_train_in - sample, dim=1)) >= self.config.loss_margin:
                    good = True
            X[i,:] = sample
        return X

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
            Xt[:,0] = torch.clip(Xt[:,0], self.domain.left, self.domain.right)
            Xt[:,1] = torch.clip(Xt[:,1], self.domain.bottom, self.domain.top)

        self.X_train_out = Xt.detach()
        self.train_loader_out = DataLoader(TensorDataset(self.X_train_out), batch_size=self.config.batch_size, shuffle=True)