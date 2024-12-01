import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.loadtxt("data_clustering.txt", delimiter=",")

num_clusters = 5

def set_plot_limits(data, padding=1):
    x_min, x_max = data[:, 0].min() - padding, data[:, 0].max() + padding
    y_min, y_max = data[:, 1].min() - padding, data[:, 1].max() + padding
    return x_min, x_max, y_min, y_max

x_min, x_max, y_min, y_max = set_plot_limits(X)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', edgecolors="black", s=80)
plt.title("Вхідні дані")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks([])
plt.yticks([])
plt.show()

kmeans = KMeans(init="k-means++", n_clusters=num_clusters, n_init=10, random_state=42)
kmeans.fit(X)

step_size = 0.01
x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

grid_points = np.c_[x_vals.ravel(), y_vals.ravel()]
output = kmeans.predict(grid_points).reshape(x_vals.shape)

plt.figure()
plt.imshow(
    output,
    interpolation='nearest',
    extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()),
    cmap=plt.cm.Paired,
    aspect='auto',
    origin="lower"
)
plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', edgecolors="black", s=80)

cluster_centers = kmeans.cluster_centers_
plt.scatter(
    cluster_centers[:, 0],
    cluster_centers[:, 1],
    marker='o',
    s=210,
    linewidth=4,
    color='black',
    facecolors='black'
)

plt.title("Границі кластерів")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks([])
plt.yticks([])
plt.show()
