from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
logistic_regression = LogisticRegression(max_iter=1000)

"""
cross validate, evaluate estimator perfomance
with logistic_regression
by iris.data and label iris.target
for 5 folds

output is estimated score of each fold(test data)
https://scikit-learn.org/stable/modules/cross_validation.html
https://towardsdatascience.com/cross-validation-explained-evaluating-estimator-performance-e51e5430ff85

ex.)
[0.96666667 1.         0.93333333 0.96666667 1.        ]
"""


"""
When target label is "organized" for specific folds number,
train data will be specialized the label or useless.

In other words...
when iris.data(label) is seperated with 3 folds,
training is executed to each labels.

---

iris.data =>
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
.
. organized format is
.
 [
  00000000000000000000000000000000000000000000000000 << fold 0
  11111111111111111111111111111111111111111111111111 << fold 1
  22222222222222222222222222222222222222222222222222 << fold 2
 ]
"""

"""
to avoid specific fold train, stratified k-fold cross validation is recommended.
https://www.datavedas.com/k-fold-cross-validation/

ex.)
cross validate score=[1.         0.83333333 1.         1.         0.93333333]
"""
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
score = cross_val_score(logistic_regression, iris.data, iris.target, cv=kfold)
print("cross validate score={}".format(score))