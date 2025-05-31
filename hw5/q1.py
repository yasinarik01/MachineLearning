import numpy as np
import matplotlib.pyplot as plt

# Generate a toy dataset: concentric circles (non-linearly separable)
from sklearn.datasets import make_circles

# Create 2D dataset with noise
X, y = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=0)

# Plot the dataset
plt.figure(figsize=(6, 6))

# Plot class 0 points in red
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')

# Plot class 1 points in blue
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')

# Add title and legend
plt.title('Non-Linearly Separable 2D Toy Dataset (Concentric Circles)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
