import numpy as np
from sklearn.model_selection import train_test_split

x = np.arange(10)
train_X, test_X = train_test_split(x, train_size=0.7, random_state=10)
print(train_X)
print(test_X)
print('--')

y = np.arange(8).reshape(2, 4).T
print(y)
print('--')

train_Y, test_Y = train_test_split(y, train_size=0.7, random_state=10)
print(train_Y)
print(test_Y)
print('--')