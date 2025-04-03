# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter

# Load the Wine dataset using the function in pandas
file_path = "C:/Users/Asus/Desktop/wine.data"
columns = [
    'Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium',
    'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins',
    'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline'
]

df = pd.read_csv(file_path, header=None, names=columns)

# Separate features and target labels
X = df.iloc[:, 1:].values  # Feature matrix
y = df.iloc[:, 0].values   # Class labels

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
# Random state is specified in fixed number. If it is 20, the numbers are considered differently
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Define Euclidean and Manhattan distance functions
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

# Implement k-NN classifier
def knn(X_train, y_train, X_test, k, distance_metric='euclidean'):
    predictions = []
    for test_point in X_test:
        if distance_metric == 'euclidean':
            distances = [euclidean_distance(test_point, train_point) for train_point in X_train]
        else:
            distances = [manhattan_distance(test_point, train_point) for train_point in X_train]
        
        # Get k nearest neighbors
        sorted_indices = np.argsort(distances)[:k]
        nearest_labels = y_train[sorted_indices]
        
        # Predict the most common class
        predicted_label = Counter(nearest_labels).most_common(1)[0][0]
        predictions.append(predicted_label)
    
    return np.array(predictions)

# Evaluate the model for different k values
k_values = [1, 3, 5, 7, 9,11,13,15,17,19,21,23,25,27,29]
accuracies_euclidean = []
accuracies_manhattan = []

results_euclidean = []
results_manhattan = []

for k in k_values:
    y_pred_euc = knn(X_train, y_train, X_test, k, distance_metric='euclidean')
    y_pred_man = knn(X_train, y_train, X_test, k, distance_metric='manhattan')
    
    acc_euc = accuracy_score(y_test, y_pred_euc)
    acc_man = accuracy_score(y_test, y_pred_man)
    
    accuracies_euclidean.append(acc_euc)
    accuracies_manhattan.append(acc_man)
    
    results_euclidean.append([k, acc_euc])
    results_manhattan.append([k, acc_man])

# Display results in tabular format
print("\nEuclidean Distance Results:")
print(pd.DataFrame(results_euclidean, columns=["K Value", "Accuracy"]))

print("\nManhattan Distance Results:")
print(pd.DataFrame(results_manhattan, columns=["K Value", "Accuracy"]))

# Plot accuracy vs. k values
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies_euclidean, marker='o', label='Euclidean')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Euclidean Distance: Model Performance vs. K Value")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies_manhattan, marker='s', label='Manhattan')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Manhattan Distance: Model Performance vs. K Value")
plt.legend()
plt.grid()
plt.show()

# Determine the best k value
best_k = k_values[np.argmax(accuracies_euclidean)]

# Generate confusion matrix and classification report for the best k
final_predictions = knn(X_train, y_train, X_test, best_k, distance_metric='euclidean')
print(f"Best K value: {best_k}")
print(confusion_matrix(y_test, final_predictions))
print(classification_report(y_test, final_predictions))

column_names = [
    "Class", "Alcohol", "Malic_acid", "Ash", "Alcalinity_of_ash", "Magnesium",
    "Total_phenols", "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins",
    "Color_intensity", "Hue", "OD280/OD315_of_diluted_wines", "Proline"
]

df = pd.read_csv(file_path, header=None, names=column_names)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df["Alcohol"], bins=20, kde=True, color="green")
plt.title("Alcohol Distribution")

plt.subplot(1, 3, 2)
sns.histplot(df["Color_intensity"], bins=20, kde=True, color="blue")
plt.title("Color Intensity Distribution")

plt.subplot(1, 3, 3)
sns.histplot(df["Proline"], bins=20, kde=True, color="red")
plt.title("Proline Distribution")

plt.tight_layout()
plt.show()

# Scatter Plot 
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Alcohol", y="Color_intensity", hue="Class", palette="viridis")
plt.title("Alcohol and Color Intensity Relation")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Alcohol", y="Proline", hue="Class", palette="coolwarm")
plt.title("Alcohol ve Proline Relation")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Color_intensity", y="Proline", hue="Class", palette="coolwarm")
plt.title("Color Intensity ve Proline Relation")
plt.show()
