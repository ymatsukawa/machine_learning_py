import numpy as np
from sklearn.preprocessing import StandardScaler

"""
standard scaler core concept is "convert data and make mean as 1, standard deviation as 0".
"""

matrix2d_X1 = [
    [1, 2],
    [3, 4],
    [5, 6]
]

matrix2d_X2 = [
    [7, 8],
    [9, 10]
]

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scaler = StandardScaler()
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler.fit
scaler.fit(matrix2d_X2)

z = scaler.transform(matrix2d_X2)

# 0.0
print(z.mean())
# 1.0
print(np.std(z))