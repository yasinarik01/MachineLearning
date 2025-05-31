import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC

# Generate non-linearly separable 2D data (concentric circles)
X, y = make_circles(n_samples=300, factor=0.5, noise=0.05)

# Define and train a linear SVM classifier
linear_svm = SVC(kernel='linear')
linear_svm.fit(X, y)

# Extract support vectors
support_vectors = linear_svm.support_vectors_

# Create a meshgrid to evaluate the decision function
x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
Z = linear_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary, margin, and regions
plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange', alpha=0.3)
plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')

# Plot the data points
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], color='red', label='Class 0')
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='blue', label='Class 1')

# Highlight the support vectors
plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
            s=100, facecolors='none', edgecolors='k', label='Support Vectors')

# Add labels and title
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Linear SVM on Non-Linearly Separable Data")
plt.legend()
plt.grid(True)
plt.show()
