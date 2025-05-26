# -*- coding: utf-8 -*-
"""
Created on Tue May 27 00:57:49 2025

@author: Asus
"""

import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train = datasets.MNIST('./data', train=True, download=True, transform=transform)
test  = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader  = DataLoader(test, batch_size=1000)

# Failed experiment variants
class BrokenCNN(nn.Module):
    def __init__(self, variant):
        super().__init__()
        layers = []
        if variant == 1:  # No activation
            layers += [nn.Conv2d(1, 16, 3, padding=1), nn.MaxPool2d(2),
                       nn.Conv2d(16, 32, 3, padding=1), nn.MaxPool2d(2)]
        elif variant == 2:  # Dropout 0.9
            layers += [nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                       nn.Dropout(0.9),
                       nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        elif variant == 3:  # LR = 1.0
            layers += [nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                       nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        self.feature = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64*7*7 if variant!=1 else 32*7*7, 10)
        )
    def forward(self, x): return self.classifier(self.feature(x))

# Training
def run_fail_experiment(variant, lr=0.001):
    device = torch.device("cpu")
    model = BrokenCNN(variant).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
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
print("Fail 1 – No Activation:", run_fail_experiment(1), "%")
print("Fail 2 – Dropout 0.9:", run_fail_experiment(2), "%")
print("Fail 3 – LR=1.0:", run_fail_experiment(3, lr=1.0), "%")
