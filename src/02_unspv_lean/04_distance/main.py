from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import os.path as p
current_dir = p.dirname(p.realpath(__file__))

np.random.seed(123)
X = np.random.random_sample([5, 3]) * 10

cols = ['X', 'Y', 'Z']
labels = ['A', 'B', 'C', 'D', 'E']
df = pd.DataFrame(X, columns=cols, index=labels)
print(df)
"""
get 3-d array with 5 rows.
ex.)
          X         Y         Z
A  6.964692  2.861393  2.268515
B  5.513148  7.194690  4.231065
C  9.807642  6.848297  4.809319
D  3.921175  3.431780  7.290497
E  4.385722  0.596779  3.980443
"""

"""
organize each row's distance.

          A         B         C         D         E
A  0.000000  4.973534  5.516653  5.899885  3.835396
B  4.973534  0.000000  4.347073  5.104311  6.698233
C  5.516653  4.347073  0.000000  7.244262  8.316594
D  5.899885  5.104311  7.244262  0.000000  4.382864
E  3.835396  6.698233  8.316594  4.382864  0.000000
"""
row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)
print(row_dist)

"""
# cluster "each row's distance" by complete linkage
# complete linkage: https://www.saedsayad.com/clustering_hierarchical.htm
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
# id-0 and id-1 is index
# id-2 is distance
# id-3 is number of original observations in the newly formed cluster.

ex.)
           label-1  label-2   distance  number of items in cluster
cluster 1      0.0      4.0   6.521973                         2.0
cluster 2      1.0      2.0   6.729603                         2.0
cluster 3      3.0      5.0   8.539247                         3.0
cluster 4      6.0      7.0  12.444824                         5.0
"""
row_clusters = linkage(row_dist, method='complete', metric='euclidean')

row_clusters_columns = ['label-1', 'label-2', 'distance', 'number of items in cluster']
row_cluster_rows = ['cluster %d' % (i + 1) for i in range(row_clusters.shape[0])]
cluster_pd = pd.DataFrame(row_clusters, columns=row_clusters_columns, index=row_cluster_rows)
print(cluster_pd)

dendro = dendrogram(row_clusters, labels=labels)
plt.savefig(p.join(current_dir, './dendrogram.png'))