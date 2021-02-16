from sklearn.svm import SVC
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=16)

"""
In order to train dataset included huge values,

1. rescale(standardize) train data
2. train with rescaled train data

otherwise, train accuracy will down.
"""

"""
standardize MinMaxScaler by follow formula

X_new_i = ( X_i - min(X) ) / ( max(X) - min(X) )

after the standardize, each value is between 0. and 1.

ex.)
[[1.364e+01 3.100e+00 2.560e+00 ... 9.600e-01 3.360e+00 8.450e+02]
 [1.260e+01 2.460e+00 2.200e+00 ... 7.300e-01 1.580e+00 6.950e+02]
 [1.196e+01 1.090e+00 2.300e+00 ... 9.900e-01 3.130e+00 8.860e+02]
 ...
 [1.242e+01 1.610e+00 2.190e+00 ... 1.060e+00 2.960e+00 3.450e+02]
 [1.390e+01 1.680e+00 2.120e+00 ... 9.100e-01 3.330e+00 9.850e+02]
 [1.416e+01 2.510e+00 2.480e+00 ... 6.200e-01 1.710e+00 6.600e+02]]

to

[[0.7016129  0.46428571 0.64516129 ... 0.35897436 0.76556777 0.43157895]
 [0.42204301 0.32983193 0.4516129  ... 0.16239316 0.11355311 0.31012146]
 [0.25       0.04201681 0.50537634 ... 0.38461538 0.68131868 0.46477733]
 ...
 [0.37365591 0.1512605  0.44623656 ... 0.44444444 0.61904762 0.02672065]
 [0.77150538 0.16596639 0.40860215 ... 0.31623932 0.75457875 0.54493927]
 [0.84139785 0.34033613 0.60215054 ... 0.06837607 0.16117216 0.28178138]]
"""
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)


"""
train data with

* X = MinMaxScaled data
* y = raw data
"""
svm = SVC()
svm.fit(X_train_scaled, y_train)
X_test_scaled = scaler.transform(X_test)

print("test score: {:.2f}".format(svm.score(X_test_scaled, y_test)))