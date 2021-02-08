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

plt.scatter(X[:, 0], X[:, 1])
plt.savefig(p.join(current_dir, './before_kmean.png'))

kmean = KMeans(n_clusters=3,
               init='random',
               max_iter=300,
               tol=1e-04,
               random_state=0)

y_kmean = kmean.fit_predict(X)

x1, y1 = X[y_kmean == 0, 0], X[y_kmean == 0, 1]
x2, y2 = X[y_kmean == 1, 0], X[y_kmean == 1, 1]
x3, y3 = X[y_kmean == 2, 0], X[y_kmean == 2, 1]

plt.scatter(x1, y1, s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1')
plt.scatter(x2, y2, s=50, c='orange', marker='o', edgecolor='black', label='cluster 2')
plt.scatter(x3, y3, s=50, c='lightblue', marker='v', edgecolor='black', label='cluster 3')

plt.legend(scatterpoints=1)
plt.savefig(p.join(current_dir, './after_kmean.png'))