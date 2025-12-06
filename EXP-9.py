import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
def plot_dendrogram(model, **kwargs):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
iris = load_iris()
X = iris.data
# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(X)
plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=3)
plt.xlabel("Samples (Index) / Cluster Sizes.")
plt.show()


Post Lab Question - 1

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X = iris.data

# Linkage methods to compare
methods = ['single', 'complete', 'average']

plt.figure(figsize=(15, 5))

for i, method in enumerate(methods):
    plt.subplot(1, 3, i+1)
    Z = linkage(X, method=method)
    dendrogram(Z, truncate_mode='level', p=3)
    plt.title(f'Linkage Method: {method.capitalize()}')
    plt.xlabel('Samples')
    plt.ylabel('Distance')

plt.tight_layout()
plt.show()

Post Lab Question - 2

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load Iris dataset
iris = load_iris()
X = iris.data

# --- Without Scaling ---
Z_original = linkage(X, method='ward')

# --- With Standardization ---
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)
Z_standardized = linkage(X_std, method='ward')

# --- With Normalization ---
scaler_norm = MinMaxScaler()
X_norm = scaler_norm.fit_transform(X)
Z_normalized = linkage(X_norm, method='ward')

# Plot comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
dendrogram(Z_original, truncate_mode='level', p=3)
plt.title('Without Scaling')

plt.subplot(1, 3, 2)
dendrogram(Z_standardized, truncate_mode='level', p=3)
plt.title('Standardized Data')

plt.subplot(1, 3, 3)
dendrogram(Z_normalized, truncate_mode='level', p=3)
plt.title('Normalized Data')

plt.tight_layout()
plt.show()
