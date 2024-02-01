import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from tqdm import tqdm
from .model import *
from .loss import *
from .callback import Callback

import mouette as M
import time

class Trainer(M.Logger):

    def __init__(self, dataset, config):
        super().__init__("Training")
        self.config = config
        self.dataset = dataset
        self.optimizer = None
        self.callbacks = []
        self.metrics = dict()
    
    def get_optimizer(self, model):
        if "adam" in self.config.optimizer.lower():
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.config.learning_rate, momentum=0.9)
        return optimizer

    def add_callbacks(self, *args):
        for cb in args:
            assert(isinstance(cb, Callback))
            self.callbacks.append(cb)

    def train(self, model):
        if self.optimizer is None :
            self.optimizer = self.get_optimizer(model)
        
        if self.config.normal_weight>0.:
            return self._train_with_normals(model)
        elif self.config.attach_weight>0.:
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
        self.metrics["test_loss"] = test_loss
        for cb in self.callbacks:
            cb.callOnEndTest(self, model)

    def _train_hkr(self, model): 
        for epoch in range(self.config.n_epochs):  # loop over the dataset multiple times
            self.metrics["epoch"] = epoch+1
            for cb in self.callbacks:
                cb.callOnBeginTrain(self, model)
            t0 = time.time()
            lossfun = LossHKR(self.config.loss_margin, self.config.loss_regul)
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
                for cb in self.callbacks:
                    cb.callOnEndForward(self, model)
            self.metrics["train_loss"] = train_loss
            self.metrics["epoch_time"] = time.time() - t0
            for cb in self.callbacks:
                cb.callOnEndTrain(self, model)
            self.evaluate_model(model)

    def _train_with_attach(self, model):
        for epoch in range(self.config.n_epochs):
            self.metrics["epoch"] = epoch+1
            for cb in self.callbacks:
                cb.callOnBeginTrain(self, model)
            t0 = time.time()
            lossfun_hkr = LossHKR(self.config.loss_margin, self.config.loss_regul)
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

                batch_loss = batch_loss_hkr + self.config.attach_weight * batch_loss_recons
                batch_loss.backward() # call back propagation
                self.optimizer.step() 
                
                for cb in self.callbacks:
                    cb.callOnEndForward(self, model)

            self.metrics["train_loss"] = [float(x) for x in (train_loss_hkr, train_loss_recons)]
            self.metrics["epoch_time"] = time.time() - t0
            for cb in self.callbacks:
                cb.callOnEndTrain(self, model)
            self.evaluate_model(model)


    def _train_with_normals(self, model):
        for epoch in range(self.config.n_epochs):
            self.metrics["epoch"] = epoch+1
            for cb in self.callbacks:
                cb.callOnBeginTrain(self, model)
            t0 = time.time()
            lossfun_hkr = LossHKR(self.config.loss_margin, self.config.loss_regul)
            train_loss_hkr = 0.
            train_loss_recons = 0.
            train_loss_normals = 0.
            for (X_in, X_out, X_bd, N_bd) in tqdm(self.dataset.train_loader, total=len(self.dataset)):
                self.optimizer.zero_grad() # zero the parameter gradients
                # forward + backward + optimize
                Y_in = model(X_in[0]) # forward computation
                Y_out = model(X_out[0])
                Y_bd = model(X_bd[0])

                ### HKR loss
                batch_loss_hkr = torch.sum(lossfun_hkr(-Y_in) + lossfun_hkr(Y_out))
                train_loss_hkr += batch_loss_hkr.detach()
                
                ### Reconstruction loss
                batch_loss_recons = torch.sum(Y_bd**2)
                train_loss_recons += batch_loss_recons.detach()

                ### Normal fitting loss
                X_bd[0].requires_grad = True
                grad = autograd.grad(outputs=model(X_bd[0]), inputs=X_bd[0],
                        grad_outputs=torch.ones_like(Y_bd).to(self.config.device),
                        create_graph=True, retain_graph=True)[0]
                batch_loss_normals = torch.sum( (grad - N_bd[0])**2)
                train_loss_normals += batch_loss_normals.detach()

                batch_loss = \
                    batch_loss_hkr \
                    + self.config.attach_weight * batch_loss_recons \
                    + self.config.normal_weight * batch_loss_normals
                batch_loss.backward() # call back propagation
                self.optimizer.step() 
                
                for cb in self.callbacks:
                    cb.callOnEndForward(self, model)

            self.metrics["train_loss"] = [float(x) for x in (train_loss_hkr, train_loss_recons, train_loss_normals)]
            self.metrics["epoch_time"] = time.time() - t0
            for cb in self.callbacks:
                cb.callOnEndTrain(self, model)
            self.evaluate_model(model)
            
