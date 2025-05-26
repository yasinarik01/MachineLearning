# -*- coding: utf-8 -*-
"""
Created on Tue May 27 00:56:54 2025

@author: Asus
"""

import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Data preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
test_loader  = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transform), batch_size=1000)

# Baseline CNN model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, 10)
        )
    def forward(self, x): return self.model(x)

device = torch.device("cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_loss, test_acc = [], []

# Training loop
for epoch in range(10):
    model.train(); total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward(); optimizer.step()
        total_loss += loss.item()
    train_loss.append(total_loss / len(train_loader))

    # Evaluation
    model.eval(); correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
    test_acc.append(100 * correct / len(test_loader.dataset))
    print(f"Epoch {epoch+1}: Loss={train_loss[-1]:.4f}, Acc={test_acc[-1]:.2f}%")

# Plotting
plt.subplot(1,2,1); plt.plot(train_loss); plt.title("Loss")
plt.subplot(1,2,2); plt.plot(test_acc); plt.title("Accuracy (%)")
plt.tight_layout(); plt.show()
