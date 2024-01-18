import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from .dataset import PointCloudDataset
from .model import *

class SH_KR:
    """ Hinge Kantorovitch-Rubinstein loss"""

    def __init__(self, margin, lbda):
        self.margin = margin  # must be small but not too small.
        self.lbda   = lbda  # must be high.

    def __call__(self, y):
        """
        Args:
            y: vector of predictions.
        """
        return  F.relu(self.margin - y) + (1./self.lbda) * torch.mean(-y)

def get_optimizer(model, config):
    if "adam" in config.optimizer.lower():
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    return optimizer

def train(model, dataset : PointCloudDataset, config):
    optimizer = get_optimizer(model, config)
    lossfun = SH_KR(config.loss_margin, config.loss_regul)
    testlossfun = nn.MSELoss() # mean square error
    for epoch in range(config.epochs):  # loop over the dataset multiple times
        train_loss = 0.
        for i, (X_in, X_out) in dataset.train_loader:
            optimizer.zero_grad() # zero the parameter gradients

            # forward + backward + optimize
            Y_in = model(X_in[0]) # forward computation
            Y_out = model(X_out[0])
            loss = torch.mean(lossfun(-Y_in) + lossfun(Y_out))
            loss.backward() # call back propagation
            train_loss += loss
            optimizer.step() 
        print(f"Train loss after epoch {epoch+1} : {train_loss}")
        test_loss = 0.
        for inputs,labels in dataset.test_loader:
            outputs = model(inputs)
            loss = testlossfun(outputs, labels)
            test_loss += config.test_batch_size * loss.item()
        print(f"Test loss after epoch {epoch+1} : {test_loss}")
        print()
