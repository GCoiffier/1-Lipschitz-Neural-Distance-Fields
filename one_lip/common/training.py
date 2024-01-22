import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from .model import *

import mouette as M

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

class Trainer(M.Logger):

    def __init__(self, dataset, config):
        super().__init__("Training")
        self.config = config
        self.dataset = dataset
        self.optimizer = None
    
    def get_optimizer(self, model):
        if "adam" in self.config.optimizer.lower():
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.config.learning_rate, momentum=0.9)
        return optimizer

    def train(self, model):
        if self.optimizer is None :
            self.optimizer = self.get_optimizer(model)
            # self.log(self.optimizer)
        if self.config.loss_attach_weight>0.:
           return self._train_with_attach(model)
        else:
            return self._train_hkr(model)

    def evaluate_model(self, model):
        """Evaluates the model on the test dataset.
        Computes the mean square error between actual distances and output of the model
        """
        test_loss = 0.
        testlossfun = nn.MSELoss() # mean square error
        for inputs,labels in self.dataset.test_loader:
            outputs = model(inputs)
            loss = testlossfun(outputs, labels)
            test_loss += self.config.test_batch_size * loss.item()
        return test_loss

    def _train_with_attach(self, model):
        lossfun_hkr = SH_KR(self.config.loss_margin, self.config.loss_regul)
        for epoch in range(self.config.epochs):
            train_loss_hkr = 0.
            train_loss_recons = 0.
            for (X_in, X_out, X_bd) in tqdm(self.dataset.train_loader, total=len(self.dataset)):
                self.optimizer.zero_grad() # zero the parameter gradients
                # forward + backward + optimize
                Y_in = model(X_in[0]) # forward computation
                Y_out = model(X_out[0])
                Y_bd = model(X_bd[0])
                batch_loss_hkr = torch.sum(lossfun_hkr(-Y_in) + lossfun_hkr(Y_out))
                train_loss_hkr += batch_loss_hkr.detach()

                batch_loss_recons = torch.sum(Y_bd**2)
                train_loss_recons += batch_loss_recons.detach()

                batch_loss = batch_loss_hkr + self.config.loss_attach_weight * batch_loss_recons
                batch_loss.backward() # call back propagation

                self.optimizer.step() 
            self.log(f"Train loss after epoch {epoch+1} : {train_loss_hkr} | {train_loss_recons}")
            self.log(f"Test loss after epoch {epoch+1} : {self.evaluate_model(model)}")
            print()

    def _train_hkr(self, model): 
        lossfun = SH_KR(self.config.loss_margin, self.config.loss_regul)
        for epoch in range(self.config.epochs):  # loop over the dataset multiple times
            train_loss = 0.
            for (X_in, X_out) in tqdm(self.dataset.train_loader, total=len(self.dataset)):
                self.optimizer.zero_grad() # zero the parameter gradients
                # forward + backward + optimize
                Y_in = model(X_in[0]) # forward computation
                Y_out = model(X_out[0])
                loss = torch.sum(lossfun(-Y_in) + lossfun(Y_out))
                loss.backward() # call back propagation
                train_loss += loss.detach()
                self.optimizer.step() 
            self.log(f"Train loss after epoch {epoch+1} : {train_loss}")
            self.log(f"Test loss after epoch {epoch+1} : {self.evaluate_model(model)}")
            print()
            
