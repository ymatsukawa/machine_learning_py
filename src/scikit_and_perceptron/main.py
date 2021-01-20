from sklearn import datasets
from sklearn.model_selection import train_test_split

"""
choose feature value and gather traing sample
"""
# gather tarininng sample No.1
## get Iris data from scikitlearn
iris = datasets.load_iris()
# feature value is chosen as inde-2 and index-3 of iris data
X = iris.data[:, [2, 3]]
# get label
y = iris.target

# gather tarininng No.2
## split data for training and will be tested
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# scale or standardize sample

# machine learn
