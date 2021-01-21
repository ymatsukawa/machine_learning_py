import os.path as p
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron
from matplotlib.colors import ListedColormap

current_dir = p.dirname(p.realpath(__file__))
data_frame = pd.read_csv(p.join(current_dir, './iris.csv'), header=None)

## target data
y = data_frame.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

## train data
X = data_frame.iloc[0:100, [0, 2]].values

perceptron = Perceptron(eta=0.1, data_num=10)
perceptron.fit(X, y)

## plot incorrect udpate by epoch
plt.plot(range(1, len(perceptron._errors) + 1), perceptron._errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

plt.savefig(p.join(current_dir, './update_num_by_epoch.png'))