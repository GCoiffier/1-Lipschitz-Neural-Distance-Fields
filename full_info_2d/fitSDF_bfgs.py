import os
import argparse
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from deel import torchlip

class SDFnet(nn.Module):

    def __init__(self):
        super().__init__()
        
        hn = 128
        self.layer1 = nn.Linear(2, hn)
        self.layer2 = nn.Linear(hn,hn)
        self.layer3 = nn.Linear(hn,hn)
        self.layer4 = nn.Linear(hn,hn//2)
        self.layer5 = nn.Linear(hn//2, 1)

    def forward(self, x):
        x = F.softplus(self.layer1(x), beta=15)
        x = F.softplus(self.layer2(x), beta=15)
        x = F.softplus(self.layer3(x), beta=15)
        x = F.softplus(self.layer4(x), beta=15)
        x = self.layer5(x)
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("name", type=str)
    parser.add_argument("-ne", "--ne", type=int, default=100, help="Number of epochs")
    args = parser.parse_args()

    #### Load dataset ####

    x_train = os.path.join("inputs", f"{args.name}_Xtrain.npy")
    y_train = os.path.join("inputs", f"{args.name}_Ytrain.npy")
    x_test = os.path.join("inputs", f"{args.name}_Xtest.npy")
    y_test = os.path.join("inputs", f"{args.name}_Ytest.npy")

    X_train = np.load(x_train)
    Y_train = np.load(y_train).reshape((X_train.shape[0],1))

    X_test = np.load(x_test)
    Y_test = np.load(y_test).reshape((X_test.shape[0], 1))

    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    X_train, Y_train = torch.Tensor(X_train).to(device), torch.Tensor(Y_train).to(device)
    X_test, Y_test = torch.Tensor(X_test).to(device), torch.Tensor(Y_test).to(device)

    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    
    train_data = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_data, batch_size=X_train.shape[0], shuffle=True)
    test_data = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_data, batch_size=X_test.shape[0])

    neural_net = SDFnet().to(device)

    # neural_net = torchlip.Sequential(
    #     torchlip.SpectralLinear(2, 256),
    #     torchlip.FullSort(),
    #     torchlip.SpectralLinear(256, 256),
    #     torchlip.FullSort(),
    #     torchlip.SpectralLinear(256, 256),
    #     torchlip.FullSort(),
    #     torchlip.SpectralLinear(256, 64),
    #     torchlip.FullSort(),
    #     torchlip.FrobeniusLinear(64, 1),
    # ).to(device)

    loss_func = nn.MSELoss() # mean square error

    #### Training ####
    optimizer = torch.optim.LBFGS(neural_net.parameters(), lr=0.3)
    for epoch in range(args.ne):  # loop over the dataset multiple times
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                output = neural_net(inputs)
                loss = loss_func(output, labels)
                loss.backward()
                return loss
            optimizer.step(closure)

        test_loss = 0.
        for inputs,labels in test_loader:
            outputs = neural_net(inputs)
            loss = loss_func(outputs, labels)
            test_loss += X_test.shape[0] * loss.item()
        print(f"Test loss after epoch {epoch+1} : {test_loss}")
        print()


    #### Output sdf as image ####
    sizeX, sizeY = 800, 800
    X = np.linspace(-0.1, 1.1, sizeX)
    Y = np.linspace(-0.1, 1.1, sizeY)
    
    img = np.zeros((sizeX,sizeY))
    for i in range(sizeX):
        inp = torch.Tensor([[X[i], Y[j]] for j in range(sizeY)]).to(device)
        img[i,:] = np.squeeze(neural_net(inp).cpu().detach().numpy())

    vmin = np.amin(img)
    vmax = np.amax(img)

    print(vmin, vmax)
    if vmin>0 or vmax<0:
        vmin,vmax = -1, 1
    
    norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
    plt.imshow(img, cmap="seismic", norm=norm)
    plt.show()