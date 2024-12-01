import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_iris

iris = load_iris()
X = iris['data']
y = iris['target']

kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

def plot_clusters(X, labels, centers=None, title="Кластери"):
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5, marker='x')
    plt.title(title)
    plt.show()

plot_clusters(X, y_kmeans, centers=kmeans.cluster_centers_, title="KMeans: 5 кластерів")

def find_clusters(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)
    initial_indices = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[initial_indices]
    
    while True:
        labels = pairwise_distances_argmin(X, centers)
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels

for rseed in [2, 0]:
    centers, labels = find_clusters(X, 3, rseed=rseed)
    plot_clusters(X, labels, centers=centers, title=f"Пошук кластерів: rseed={rseed}")

kmeans_3 = KMeans(n_clusters=3, random_state=0)
labels_3 = kmeans_3.fit_predict(X)
plot_clusters(X, labels_3, centers=kmeans_3.cluster_centers_, title="KMeans: 3 кластери")
