import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

X = np.loadtxt("data_clustering.txt", delimiter=",")

bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

cluster_centers = meanshift_model.cluster_centers_
print("Cluster centers:\n", cluster_centers)

labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print("\nNumber of clusters in input data:", num_clusters)

def plot_meanshift_clusters(X, labels, cluster_centers, title="Кластери"):
    plt.figure()
    colors = cycle('bgrcmyk')
    markers = cycle('o*xvs^')

    for cluster_idx, color, marker in zip(range(num_clusters), colors, markers):
        cluster_points = X[labels == cluster_idx]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], marker=marker, color=color, label=f"Cluster {cluster_idx + 1}")

        cluster_center = cluster_centers[cluster_idx]
        plt.plot(cluster_center[0], cluster_center[1], marker='o', markersize=15,
                 markerfacecolor=color, markeredgecolor='black')

    plt.title(title)
    plt.legend()
    plt.show()

plot_meanshift_clusters(X, labels, cluster_centers)
