import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn import svm
import os.path as p
current_dir = p.dirname(p.realpath(__file__))

# 2x500 matrix. value is random
matrix = np.random.randn(500, 2)

"""
calculate xor by each row, and get 1 or -1 convert from True or False
[X1 xor Y1, X2 xor Y2, ..., Xn xor Yn]
#=> ex.) [1, -1, ..., 1]
"""
xor = np.logical_xor(matrix[:, 0] > 0, matrix[:, 1] > 0)
xor = np.where(xor, 1, -1)

x1 = matrix[xor == 1, 0]
y1 = matrix[xor == 1, 1]
plt.scatter(x1, y1, c='b', marker='x', label='1')

x2 = matrix[xor == -1, 0]
y2 = matrix[xor == -1, 1]
plt.scatter(x2, y2, c='r', marker='x', label='-1')
plt.legend(loc='upper left')

plt.savefig(p.join(current_dir, './svm_non_linear_xor.png'))
plt.cla()

# non linear svm
X = matrix
y = xor

svm = svm.SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X, y)

plot_decision_regions(X, y, clf=svm)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(p.join(current_dir, './svm_non_linear_xor_with_decision_regions.png'))