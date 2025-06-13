import torch
import numpy as np

class Linear:
    def __init__(self, nin, nout, device="cpu"):
        self.W = torch.randn(nin, nout, device=device, requires_grad=False) * np.sqrt(2.0 / nin)
        self.b = torch.zeros(nout, device=device, requires_grad=False)

    def forward(self, X):
        self.X = X
        Z = torch.matmul(self.X, self.W) + self.b
        return Z

    def backward(self, dZ):
        self.dZ = dZ
        self.dW = torch.matmul(self.X.T, self.dZ)
        self.db = torch.sum(dZ, dim = 0)
        self.dX = torch.matmul(self.dZ, self.W.T)
        return self.dX #What the net need to backpropagate

    def update(self, lr):
        self.W = self.W - (lr * self.dW)
        self.b = self.b - (lr*self.db)

class CrossEntropyFromLogits:
    def forward(self, Z, Y):
        self.Y = Y
        self.A = torch.nn.functional.softmax(Z, dim=1)
        log_softmax_Z = torch.nn.functional.log_softmax(Z, dim=1)
        log_probs = log_softmax_Z[torch.arange(Z.size(0)), Y] #(n_batch,)
        loss = -torch.mean(log_probs, dim = 0)
        return loss

    def backward(self, n_classes):
        Y_one_hot = torch.nn.functional.one_hot(self.Y, num_classes=n_classes)
        batch_size = self.Y.shape[0]
        dZ = (self.A - Y_one_hot)/batch_size
        return dZ

class Net:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dZ):
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)
        return dZ

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

class ReLU:
    def forward(self, Z):
        self.Z = Z
        self.A = self.Z * (self.Z > 0)
        return self.A

    def backward(self, dA):
        dZ = dA * (self.Z > 0)
        return dZ

    def update(self,lr):
        pass
