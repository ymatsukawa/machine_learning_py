from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import os.path as p
from sklearn.cluster import KMeans
current_dir = p.dirname(p.realpath(__file__))

X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

# when you don't know what cluster size is best.

distortions = []
cluster_nums = 10
for cluster_num in range(1, cluster_nums):
    kmean = KMeans(n_clusters=cluster_num,
                   init='k-means++')
    kmean.fit(X)
    distortions.append(kmean.inertia_)

plt.xlabel('cluster num')
plt.ylabel('distortions')
plt.plot(range(1, cluster_nums), distortions, marker='o')
plt.savefig(p.join(current_dir, './elbow.png'))