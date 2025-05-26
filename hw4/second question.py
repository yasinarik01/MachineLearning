# -*- coding: utf-8 -*-
"""
Created on Tue May 27 00:57:11 2025

@author: Asus
"""

import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
test_loader  = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transform), batch_size=1000)

# Variant A: Deeper network
class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(128*7*7, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)

# Variant B: Kernel sizes 5x5 and 1x1
class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)

# Variant C: LeakyReLU + BatchNorm
class ModelC(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.LeakyReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.LeakyReLU(),
            nn.Dropout(0.5), nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)

def train_and_eval(ModelClass):
    device = torch.device("cpu")
    model = ModelClass().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.001)
    for _ in range(3):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
    model.eval(); correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    return 100 * correct / len(test_loader.dataset)

# Output
print("Model A (Deeper):", train_and_eval(ModelA), "%")
print("Model B (1x1/5x5):", train_and_eval(ModelB), "%")
print("Model C (LeakyReLU + BatchNorm):", train_and_eval(ModelC), "%")
