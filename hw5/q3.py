import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC

# Generate non-linearly separable data
X, y = make_circles(n_samples=300, factor=0.5, noise=0.05)

# Train SVM with RBF kernel
svm_rbf = SVC(kernel='rbf', C=1, gamma='auto')
svm_rbf.fit(X, y)

# Get support vectors
support_vectors = svm_rbf.support_vectors_

# Create meshgrid for decision boundary
x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
Z = svm_rbf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary and margins
plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, Z > 0, alpha=0.3, cmap='coolwarm')
plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')

# Plot data points
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', label='Class 1')

# Plot support vectors
plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
            s=100, facecolors='none', edgecolors='black', label='Support Vectors')

# Label plot
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("2D Decision Boundary with RBF Kernel SVM")
plt.legend()
plt.grid(True)
plt.show()
