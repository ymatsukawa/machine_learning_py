import numpy as np
import pandas as pd
import os.path as p
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
current_dir = p.dirname(p.realpath(__file__))

data_frame = pd.read_csv(p.join(current_dir, '../data/fish.csv'), header=None, skiprows=1)

# preparation
data = data_frame[data_frame[0] == 'Bream']
data = data_frame.iloc[:, [6, 1]].values

## get width[index = 6] and weight[index = 1]
X = data_frame.iloc[:, [6]].values[1:]
y = data_frame.iloc[:, [1]].values[1:]

# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y)

# plots from split data
plt.scatter(X_train, y_train, s=15, c='b')
plt.xlabel("bream-width")
plt.ylabel("bream-weight")
plt.savefig(p.join(current_dir, './train_width-weight.png'))

plt.cla()

plt.scatter(X_test, y_test, s=15, c='r')
plt.xlabel("bream-width")
plt.ylabel("bream-weight")
plt.savefig(p.join(current_dir, './test_width-weight.png'))

# linear regression and measure score
linear_regression = LinearRegression().fit(X_train, y_train)

## y = wx + b
### coef_ (coefficient) is w
print("learned coefficient: {}".format(linear_regression.coef_))
### intercept_ is b
print("learned intercept: {}".format(linear_regression.intercept_))

## measure learning score
print("training score: {:.2f}".format(linear_regression.score(X_test, y_test)))
print("test score: {:.2f}".format(linear_regression.score(X_train, y_train)))