from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import os.path as p
current_dir = p.dirname(p.realpath(__file__))

X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

# when you don't know what cluster size is best.

# 3 cluster
range_n_clusters = [2, 3]
# 6 cluster
# range_n_clusters = [2, 3, 4, 5, 6]

y_lower = 10
y_ticks = []
for n_clusters in range_n_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    cluster = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = cluster.fit_predict(X)

    silhoulette_avg = silhouette_score(X, cluster_labels)
    print("For cluster num = ", n_clusters, "The avg silhouette score is ", silhoulette_avg)

    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    for index in range(n_clusters):
        i_th_silhouette_values = sample_silhouette_values[cluster_labels == index]
        i_th_silhouette_values.sort()

        i_th_cluster_size = i_th_silhouette_values.shape[0]

        y_upper = y_lower + len(i_th_silhouette_values)
        color = cm.jet(float(index) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                 0,
                 i_th_silhouette_values,
                 alpha=0.7,
                 edgecolor='none',
                 facecolor=color)
        y_lower = y_upper

    
    ax1.set_ylabel('Cluster')
    ax1.set_xlabel('Silhouette Cofficient')
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6])

    # plot scatter on axis 2
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=100, lw=0, alpha=0.7, c=colors, edgecolor='k')

    # plot cluster center on axis 2
    centers = cluster.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c='white', alpha=1, s=200, edgecolor='k')

plt.savefig(p.join(current_dir, './silhouette.png'))
