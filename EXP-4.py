# Import necessary modules
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay

# Loading data
iris = load_iris(as_frame=True)

# Create feature and target arrays
X = iris.data[["sepal length (cm)", "sepal width (cm)"]]
y = iris.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# Creating a Pipeline
clf = Pipeline(steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=11))])

# Creates a figure with two subplots (side by side)
_, axs = plt.subplots(ncols=2, figsize=(12, 5))

# Visualizing the Classifier
for ax, weights in zip(axs, ("uniform", "distance")):
    # Set the weights parameter and fit the model
    clf.set_params(knn__weights=weights).fit(X_train, y_train)

    # Create decision boundary display
    disp = DecisionBoundaryDisplay.from_estimator(
        clf, X_train, # Changed from X_test to X_train for proper decision boundary
        response_method="predict",
        plot_method="pcolormesh",
        xlabel=iris.feature_names[0],
        ylabel=iris.feature_names[1],
        shading="auto",
        alpha=0.5,
        ax=ax
    )

    # Scatter plot of all data points
    scatter = disp.ax_.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors="k")

    # Add legend
    disp.ax_.legend(
        scatter.legend_elements()[0],
        iris.target_names,
        loc="lower left",
        title="Classes"
    )

    # Set title (fixed string formatting)
    disp.ax_.set_title(f"3-Class classification\n(k={clf[-1].n_neighbors}, weights='{weights}')")

plt.tight_layout()
plt.show()

# Additional: Print model accuracy for both weight strategies
print("Model Performance:")
print("-" * 30)

for weights in ("uniform", "distance"):
    clf.set_params(knn__weights=weights).fit(X_train, y_train)
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    print(f"Weights: {weights}")
    print(f" Training Accuracy: {train_accuracy:.4f}")
    print(f" Test Accuracy: {test_accuracy:.4f}")
    print()

# Bonus: Analysis of different k values
print("K-value Analysis:")
print("-" * 30)

k_values = range(1, 21)
train_accuracies_uniform = []
test_accuracies_uniform = []
train_accuracies_distance = []
test_accuracies_distance = []

for k in k_values:
    # Uniform weights
    clf_uniform = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights="uniform"))
    ])
    clf_uniform.fit(X_train, y_train)
    train_accuracies_uniform.append(clf_uniform.score(X_train, y_train))
    test_accuracies_uniform.append(clf_uniform.score(X_test, y_test))

    # Distance weights
    clf_distance = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance"))
    ])
    clf_distance.fit(X_train, y_train)
    train_accuracies_distance.append(clf_distance.score(X_train, y_train))
    test_accuracies_distance.append(clf_distance.score(X_test, y_test))

# Plot accuracy vs k
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, train_accuracies_uniform, 'b-o', label='Training (Uniform)', markersize=4)
plt.plot(k_values, test_accuracies_uniform, 'b--s', label='Test (Uniform)', markersize=4)
plt.plot(k_values, train_accuracies_distance, 'r-o', label='Training (Distance)', markersize=4)
plt.plot(k_values, test_accuracies_distance, 'r--s', label='Test (Distance)', markersize=4)
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs k Value')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(k_values, test_accuracies_uniform, 'b-o', label='Uniform Weights', markersize=4)
plt.plot(k_values, test_accuracies_distance, 'r-s', label='Distance Weights', markersize=4)
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy Comparison: Uniform vs Distance Weights')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Find optimal k value
optimal_k_uniform = k_values[np.argmax(test_accuracies_uniform)]
optimal_k_distance = k_values[np.argmax(test_accuracies_distance)]

print(f"Optimal k for Uniform weights: {optimal_k_uniform} (Accuracy: {max(test_accuracies_uniform):.4f})")
print(f"Optimal k for Distance weights: {optimal_k_distance} (Accuracy: {max(test_accuracies_distance):.4f})")
