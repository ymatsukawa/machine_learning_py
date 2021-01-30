import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
import os.path as p
current_dir = p.dirname(p.realpath(__file__))

SEP_LENGTH='sepal length (cm)'
SEP_WIDTH='sepal width (cm)'
iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

data = iris_df.loc[:, [SEP_LENGTH, SEP_WIDTH]]
X = data.loc[:].values
Y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.6, random_state=1)

# plot raw train data
plt.scatter(X_train[:, 0], X_train[:, 1], s=15, c='b')
plt.xlabel(SEP_LENGTH)
plt.ylabel(SEP_WIDTH)
plt.savefig(p.join(current_dir, './train_iris_raw.png'))
plt.cla()

classification = svm.SVC()
classification.fit(X_train, y_train)

# plot with decision regions
plot_decision_regions(X_train, y_train, clf=classification)
plt.xlabel(SEP_LENGTH)
plt.ylabel(SEP_WIDTH)
plt.savefig(p.join(current_dir, './train_iris_with_decision_regions.png'))

predict = classification.predict(X_test)
"""
array of predicted labels about X_test(x-y points)
"""
print(predict)