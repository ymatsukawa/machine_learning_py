import numpy as np
import pandas as pd
import os.path as p
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions

current_dir = p.dirname(p.realpath(__file__))
def target_file(target_file_relative_path):
    return p.join(p.dirname(p.realpath(__file__)), target_file_relative_path)

def income_label(row):
    age = row.age
    income = row.fnlwgt
    middle_inc = 20 * 10000
    high_inc = 50 * 10000
    if(0 <= income and income <= middle_inc):
        return 0 # 'challenger'
    elif(middle_inc < income and income <= high_inc):
        return 1 # 'world manager'
    else:
        if(age <= 30):
            return 2 # 'reborn'
        elif(30 < age and age <= 60):
            return 3 # 'strategist'
        else:
            return 4 # 'waiting new age'

df = pd.read_csv(target_file('../data/income.csv'))

FINAL_WEIGHT = 2
AGE = 0
LABEL_COL = 'income_state'

data = df.iloc[:, [AGE, FINAL_WEIGHT]].values
X = data[:, 0]
y = data[:, 1]
df.loc[:, LABEL_COL] = df.apply(income_label, axis=1)

# plots
plt.xlabel("age")
plt.ylabel("final income weight")
plt.scatter(X, y, marker='x', c='b')
plt.savefig(p.join(current_dir, './age-incomeweight.png'))
plt.cla()

labels = df.loc[:, LABEL_COL].values
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data, labels)

plot_decision_regions(data, labels, clf=knn)
plt.xlabel('age')
plt.ylabel('final income weight')
plt.savefig(p.join(current_dir, './age-incomeweight_with_decision_regions.png'))