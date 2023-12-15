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
            "train" : os.path.join("inputs", f"{name}_Xtrain.npy"),
            "Xtest" : os.path.join("inputs", f"{name}_Xtest.npy"),
            "Ytest" : os.path.join("inputs", f"{name}_Ytest.npy")
        }
        self.config = config
        self.domain = [-0.6,1.6]
        
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
        return (float(vmin[0]), float(vmax[0])), (float(vmin[1]), float(vmax[1]))

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
        self.X_train_out = self.generate_complementary_distribution(self.X_train_in)

        self.log(f"Found {self.X_train_in.shape[0]} training examples in dataset")
        self.log(f"Generated {self.X_train_out.shape[0]} training examples out of distribution")

        self.X_train_in = torch.Tensor(self.X_train_in).to(self.config.device)
        self.train_loader_in = DataLoader(TensorDataset(self.X_train_in), batch_size=self.config.batch_size, shuffle=True)

        self.X_train_out = torch.Tensor(self.X_train_out).to(self.config.device)
        self.train_loader_out = DataLoader(TensorDataset(self.X_train_out), batch_size=self.config.batch_size, shuffle=True)

        test_data = TensorDataset(self.X_test, self.Y_test)
        self.test_loader = DataLoader(test_data, batch_size=self.config.test_batch_size)
        self.log("...Dataset loading complete")

    def generate_complementary_distribution_hole(self, X):
        n_pts = X.shape[0]
        EPS = 0.01
        xmin,ymin = np.amin(X, axis=0)
        xmax,ymax = np.amax(X, axis=0)
        X2 = np.array([-0.2,-0.2]) + 1.5*np.random.random((2*n_pts, 2))
        I = np.ones(X2.shape[0], dtype=bool)
        for i in range(X2.shape[0]):
            x,y = X2[i,:]
            if xmin - EPS <= x <= xmax + EPS and ymin - EPS <= y <= ymax + EPS:
                I[i] = False # reject point if inside bounding box
        X2 = X2[I,:]
        return X2[:n_pts, :]

    def generate_complementary_distribution(self, X):
        n_pts = X.shape[0]
        domain_width = self.domain[1] - self.domain[0]
        X2 = domain_width*np.random.random((n_pts,2)) + self.domain[0]
        return X2

    def update_complementary_distribution(self, model, maxiter, level_set=-1e-3):
        """
        Perform Newton-Raphson iteration on the points to make them closer

        Args:
            sdf: pytorch model
        """
        step_size = 1. / maxiter
        learning_rate = 0.5
        Xt = self.X_train_out.clone()

        for step in range(maxiter):

            Xt = Xt.detach()
            Xt.requires_grad = True

            # Compute signed distances
            y = model(Xt)
            Gy = torch.ones_like(y)
            y.backward(Gy,retain_graph=True)
            
            # retrieve gradient of the function
            grad = Xt.grad
            grad_norm_squared = torch.sum(grad**2, axis=1).reshape((5000,1))            
            grad = grad / (grad_norm_squared + 1e-8)
            target = -y + level_set
            # target = F.leaky_relu(-y + level_set, 0.2)

            Xt = Xt + learning_rate * step_size * target * grad
            
            # clipping to domain
            Xt = torch.clip(Xt, self.domain[0], self.domain[1])

        self.X_train_out = Xt.detach()
        self.train_loader_out = DataLoader(TensorDataset(self.X_train_out), batch_size=self.config.batch_size, shuffle=True)