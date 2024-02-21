import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from tqdm import tqdm
from .loss import *
from .callback import Callback

import mouette as M
import time

class Trainer(M.Logger):

    def __init__(self, train_loaders, test_loader, config):
        super().__init__("Training")
        self.config = config
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.optimizer = None
        self.scheduler = None
        self.callbacks = []
        self.metrics = dict()
    
    def get_optimizer(self, model):
        if "adam" in self.config.optimizer.lower():
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.config.learning_rate, momentum=0.9)
        return optimizer
    
    def get_scheduler(self):
        if self.optimizer is None : return None
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)

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

    def train_lip(self, model):
        self.optimizer = self.get_optimizer(model)
        self.scheduler = self.get_scheduler()
        for epoch in range(self.config.n_epochs):
            self.metrics["epoch"] = epoch+1
            for cb in self.callbacks:
                cb.callOnBeginTrain(self, model)
            t0 = time.time()
            lossfun_hkr = LossHKR(self.config.loss_margin, self.config.loss_regul)
            train_loss = dict()
            train_loss["hkr"] = 0.
            if self.config.attach_weight >0. :
                train_loss["recons"] = 0.
            if self.config.normal_weight >0. :
                train_loss["normals"] = 0.
            if self.config.eikonal_weight >0. :
                train_loss["eik"] = 0.
                ones = torch.ones(self.config.batch_size).to(self.config.device)
            if self.config.gnorm_weight > 0.:
                train_loss["gnorm"] = 0.
            
            train_length = len(self.train_loaders[0])
            for (X_in, X_out, X_bd) in tqdm(zip(*self.train_loaders), total=train_length):
                self.optimizer.zero_grad() # zero the parameter gradients
                # forward + backward + optimize
                if self.config.normal_weight>0.:
                    X_bd, N_bd = X_bd
                    X_bd.requires_grad = True
                else:
                    X_bd, = X_bd
                Y_in = model(X_in[0]) # forward computation
                Y_out = model(X_out[0])
                Y_bd = model(X_bd)
                total_batch_loss = 0.

                ### HKR loss
                batch_loss_hkr = torch.sum(lossfun_hkr(-Y_in) + lossfun_hkr(Y_out))
                train_loss["hkr"] += float(batch_loss_hkr.detach())
                total_batch_loss += batch_loss_hkr
                
                ### Normal fitting loss
                if self.config.normal_weight>0.:
                    grad = autograd.grad(outputs=Y_bd, inputs=X_bd,
                            grad_outputs=torch.ones_like(Y_bd).to(self.config.device),
                            create_graph=True, retain_graph=True)[0]
                    batch_loss_normals = self.config.normal_weight*vector_alignment_loss(grad, N_bd[0])
                    total_batch_loss += batch_loss_normals
                    train_loss["normals"] += float(batch_loss_normals.detach())
               
                ### Reconstruction loss
                if self.config.attach_weight>0.:
                    batch_loss_recons = self.config.attach_weight * torch.sum(Y_bd**2)
                    train_loss["recons"] += float(batch_loss_recons.detach())
                    total_batch_loss += batch_loss_recons

                ### Eikonal loss
                if self.config.eikonal_weight>0.:
                    x_rdm = 2*torch.rand_like(X_in[0])-1 # between -1 and 1
                    x_rdm.requires_grad = True
                    y_rdm = model(x_rdm)
                    batch_grad = torch.autograd.grad(y_rdm, x_rdm, grad_outputs=torch.ones_like(y_rdm), create_graph=True)[0]
                    # batch_loss_eik = self.config.eikonal_weight * F.mse_loss(torch.sum(batch_grad*batch_grad, axis=-1), ones)
                    batch_loss_eik = self.config.eikonal_weight * F.mse_loss(batch_grad.norm(dim=1), ones)
                    total_batch_loss += batch_loss_eik
                    train_loss["eik"] += float(batch_loss_eik.detach())

                ### Max grad norm loss
                if self.config.gnorm_weight>0.:
                    x_rdm = 2*torch.rand_like(X_in[0])-1 # between -1 and 1
                    x_rdm.requires_grad = True
                    y_rdm = model(x_rdm)
                    batch_grad = torch.autograd.grad(y_rdm, x_rdm, grad_outputs=torch.ones_like(y_rdm), create_graph=True)[0]
                    # batch_loss_gnorm = -self.config.gnorm_weight * torch.sum(batch_grad.norm(dim=1))
                    batch_loss_gnorm = self.config.gnorm_weight * (1-torch.sum(batch_grad * batch_grad))
                    total_batch_loss += batch_loss_gnorm
                    train_loss["gnorm"] += float(batch_loss_gnorm.detach()) 

                total_batch_loss.backward() # call back propagation
                self.optimizer.step() 
                for cb in self.callbacks:
                    cb.callOnEndForward(self, model)
            
            if self.scheduler is not None:
                self.scheduler.step()

            self.metrics["train_loss"] = train_loss
            self.metrics["epoch_time"] = time.time() - t0
            for cb in self.callbacks:
                cb.callOnEndTrain(self, model)
            self.evaluate_model(model)
            
    def train_lip_unsigned(self, model):
        self.optimizer = self.get_optimizer(model)
        for epoch in range(self.config.n_epochs):
            self.metrics["epoch"] = epoch+1
            for cb in self.callbacks:
                cb.callOnBeginTrain(self, model)
            t0 = time.time()
            lossfun_hkr = LossHKR(self.config.loss_margin, self.config.loss_regul)
            train_loss = dict()
            train_loss["hkr"] = 0.
            train_length = len(self.train_loaders[0])
            for (X_on, X_out) in tqdm(zip(*self.train_loaders), total=train_length):
                self.optimizer.zero_grad() # zero the parameter gradients
                # forward + backward + optimize
                Y_on = model(X_on[0]) # forward computation
                Y_out = model(X_out[0])
                ### HKR loss
                batch_loss_hkr = torch.sum(lossfun_hkr(-Y_on) + lossfun_hkr(Y_out))
                train_loss["hkr"] += float(batch_loss_hkr.detach())
                batch_loss_hkr.backward() # call back propagation
                self.optimizer.step() 
                for cb in self.callbacks:
                    cb.callOnEndForward(self, model)
            self.metrics["train_loss"] = train_loss
            self.metrics["epoch_time"] = time.time() - t0
            for cb in self.callbacks:
                cb.callOnEndTrain(self, model)
            self.evaluate_model(model)

    def train_full_info(self, model):
        self.optimizer = self.get_optimizer(model)   
        for epoch in range(self.config.n_epochs):  # loop over the dataset multiple times
            self.metrics["epoch"] = epoch+1
            for cb in self.callbacks:
                cb.callOnBeginTrain(self, model)
            t0 = time.time()
            train_loss = dict()
            train_loss["fit"] = 0.
            if self.config.eikonal_weight >0. :
                train_loss["eik"] = 0.
            if self.config.tv_weight >0. :
                train_loss["tv"] = 0.

            for (x,y_target) in tqdm(self.train_loaders, total=len(self.train_loaders)):
                self.optimizer.zero_grad() # zero the parameter gradients
                # forward + backward + optimize
                x.requires_grad = True
                y_pred = model(x) # forward computation
                total_batch_loss = 0.

                ### Fitting loss
                batch_loss_fit = F.mse_loss(y_pred, y_target)
                train_loss["fit"] += batch_loss_fit.item()
                total_batch_loss += batch_loss_fit

                ### Eikonal loss
                if self.config.eikonal_weight>0.:
                    x_rdm = 2*torch.rand_like(x)-1 # between -1 and 1
                    x_rdm.requires_grad = True
                    y_rdm = model(x_rdm)
                    batch_grad = torch.autograd.grad(y_rdm, x_rdm, grad_outputs=torch.ones_like(y_rdm), create_graph=True)[0]
                    # batch_loss_eik = self.config.eikonal_weight * F.mse_loss(torch.sum(batch_grad*batch_grad, axis=-1), ones)
                    batch_grad_norm = batch_grad.norm(dim=-1)
                    batch_loss_eik = self.config.eikonal_weight * F.mse_loss(batch_grad_norm, torch.ones_like(batch_grad_norm))
                    total_batch_loss += batch_loss_eik
                    train_loss["eik"] += float(batch_loss_eik.detach())

                total_batch_loss.backward() # call back propagation
                self.optimizer.step()
                for cb in self.callbacks:
                    cb.callOnEndForward(self, model)
            self.metrics["train_loss"] = train_loss
            self.metrics["epoch_time"] = time.time() - t0
            for cb in self.callbacks:
                cb.callOnEndTrain(self, model)
            self.evaluate_model(model)

    def train_sal(self, model):
        self.optimizer = self.get_optimizer(model)
        sal_lossfun = SALLoss(1., self.config.metric)

        for epoch in range(self.config.n_epochs):  # loop over the dataset multiple times
            self.metrics["epoch"] = epoch+1
            for cb in self.callbacks:
                cb.callOnBeginTrain(self, model)
            t0 = time.time()
            train_loss = dict()
            train_loss["sal"] = 0.
            if self.config.attach_weight >0.:
                train_loss["fit"] = 0.

            train_length = len(self.train_loaders[0])
            for ((x_out,y_target), x_on) in tqdm(zip(*self.train_loaders), total=train_length):
                self.optimizer.zero_grad() # zero the parameter gradients
                # forward + backward + optimize
                x_out.requires_grad = True
                x_on.requires_grad = True
                total_batch_loss = 0.

                ### SAL loss
                y_pred = model(x_out) # forward computation
                batch_loss_sal = sal_lossfun(y_pred, y_target)
                train_loss["sal"] += float(batch_loss_sal.detach())
                total_batch_loss += batch_loss_sal

                ### Fitting loss
                if self.config.attach_weight>0.:
                    y_pred0 = model(x_on)
                    batch_loss_fit = self.config.attach_weight * torch.sum(y_pred0**2)
                    train_loss["fit"] += batch_loss_fit.item()
                    total_batch_loss += batch_loss_fit

                total_batch_loss.backward() # call back propagation
                self.optimizer.step()
                for cb in self.callbacks:
                    cb.callOnEndForward(self, model)
            self.metrics["train_loss"] = train_loss
            self.metrics["epoch_time"] = time.time() - t0
            for cb in self.callbacks:
                cb.callOnEndTrain(self, model)
            self.evaluate_model(model)