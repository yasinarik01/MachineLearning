# -*- coding: utf-8 -*-
"""
Created on Tue May 27 00:57:37 2025

@author: Asus
"""

import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST('./data', train=False, download=True, transform=transform)
def get_loader(batch_size): return DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# Model
class FlexibleCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)

# Weight initialization
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

# Training and evaluation
def train_and_eval(opt_name="adam", lr=0.001, dropout=0.5, batch_size=64, init=False):
    device = torch.device("cpu")
    model = FlexibleCNN(dropout).to(device)
    if init: model.apply(init_weights)
    train_loader = get_loader(batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr) if opt_name=="adam" else optim.SGD(model.parameters(), lr)
    for _ in range(3):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
    model.eval(); correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    return 100 * correct / len(test_loader.dataset)

# Output
print("Model 1 (Adam, 0.001, drop=0.5):", train_and_eval("adam", 0.001, 0.5), "%")
print("Model 2 (SGD, 0.01, drop=0.2):", train_and_eval("sgd", 0.01, 0.2), "%")
print("Model 3 (Adam, 0.0001, drop=0.0, init):", train_and_eval("adam", 0.0001, 0.0, 64, True), "%")
