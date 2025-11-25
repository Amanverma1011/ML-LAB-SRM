import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
# create dataset
X, y = make_blobs(n_samples=150, n_features=2,centers=3,
cluster_std=0.5,shuffle=True, random_state=0)
# Visualize data
plt.scatter(X[:, 0], X[:, 1],c='white', marker='o',edgecolor='black',s=50)
plt.show()
# Initialize KMeans
km = KMeans(n_clusters=3, init='random',n_init=10, max_iter=300,tol=1e-04,
random_state=0)
# Fit and Predict
y_km = km.fit_predict(X)
# Visualize clusters
plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1],s=50, c='lightgreen',marker='s',
edgecolor='black',label='cluster 1')

plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1],s=50, c='orange',marker='o',
edgecolor='black',label='cluster 2')
plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1],s=50, c='lightblue',marker='v',
edgecolor='black',label='cluster 3')
# plot the centroids
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],s=250, marker='*',c='red',
edgecolor='black',label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.show()


Post Lab - 1

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay

# Create moons dataset
X, y = make_moons(n_samples=200, noise=0.25, random_state=0)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN Classifier
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train_scaled, y_train)

# Plot decision boundary
disp = DecisionBoundaryDisplay.from_estimator(
    clf, X_train_scaled, response_method="predict",
    plot_method="pcolormesh", shading="auto"
)
disp.ax_.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, edgecolor="k")
plt.title("KNN Classification on Make Moons Dataset")
plt.show()

# Accuracy
print(f"Train Accuracy: {clf.score(X_train_scaled, y_train):.2f}")
print(f"Test Accuracy: {clf.score(X_test_scaled, y_test):.2f}")


Post Lab - 2

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Create dataset with non-uniform cluster sizes
X, y = make_blobs(
    n_samples=[100, 300, 600],   # non-uniform sizes
    centers=[[0,0], [5,5], [10,0]],
    cluster_std=[0.5, 1.5, 1.0],
    random_state=42
)

# Apply KMeans
km = KMeans(n_clusters=3, random_state=0)
y_km = km.fit_predict(X)

# Plot clusters
plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s=50, c='lightgreen', label='Cluster 1')
plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=50, c='orange', label='Cluster 2')
plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s=50, c='lightblue', label='Cluster 3')

# Plot centroids
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            s=250, marker='*', c='red', label='Centroids')
plt.title("KMeans on Non-uniform Cluster Sizes")
plt.legend()
plt.show()

