import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

# Create a dataset manually (the Iris dataset is typically 150 samples with 4 features)
data = {
    "sepal_length": [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1],
    "sepal_width": [3.5, 3.0, 3.2, 3.4, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3.4, 3.0, 4.0, 4.1, 3.9, 3.5, 3.8, 3.8],
    "petal_length": [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.3, 1.5, 1.7, 1.6, 1.5, 1.6, 1.2, 1.3, 1.4, 1.7, 1.7, 1.5],
    "petal_width": [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.3, 0.3],
    "species": ["setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", "setosa", 
                "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor", "versicolor"]
}
df = pd.DataFrame(data)

print(df)

# Step 1: Load the dataset from the CSV file
df = pd.read_csv('iris_flower_data.csv')

# Step 2: Preprocess the data
X = df.drop(columns=['species'])
y = df['species']

# Convert the target labels into numeric form (Setosa -> 0, Versicolor -> 1, Virginica -> 2)
y = y.map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

# Step 3: Visualize the distribution of the features (using histograms)
plt.figure(figsize=(12, 8))
df.drop(columns=['species']).hist(bins=20, figsize=(12, 8), color='skyblue', edgecolor='black')
plt.suptitle("Distribution of Iris Flower Features", fontsize=16)
plt.show()

# Step 4: Plot pairwise relationships (using Seaborn's pairplot)
sns.pairplot(df, hue='species', palette='Set2', markers=["o", "s", "D"])
plt.suptitle("Pairwise Relationships between Features", fontsize=16)
plt.show()

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Initialize the classifier (K-Nearest Neighbors)
model = KNeighborsClassifier(n_neighbors=5)

# Step 8: Train the model
model.fit(X_train, y_train)

# Step 9: Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Step 10: Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm, display_labels=['Setosa', 'Versicolor', 'Virginica'])
cm_display.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Step 11: Plot decision boundaries (using PCA for dimensionality reduction to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Train the model on PCA-transformed data for plotting decision boundary
model_pca = KNeighborsClassifier(n_neighbors=5)
model_pca.fit(X_pca, y)

# Create a meshgrid for plotting decision boundaries
h = .02  # step size in the mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict on the grid to create the decision boundary plot
Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting the decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set2')
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolor='k', cmap='Set2', marker='o', s=50, label="Data Points")
plt.title("KNN Decision Boundaries (PCA reduced)", fontsize=16)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()
