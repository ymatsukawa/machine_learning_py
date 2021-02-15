from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

"""
grid search by manually
human should create loop.
"""
search_range = [0.001, 0.01, 0.1, 1, 10, 100]
best_score = 0
best_parameters = {}

for gamma in search_range:
    for C in search_range:
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}

print("Best score {:.2f}".format(best_score))
print("Best parameters: {}".format(best_parameters))
print('---------------')

"""
grid search by library.
human only input evaluation value.
"""

search_range = [0.001, 0.01, 0.1, 1, 10, 100]
svc_params = {'C': search_range, 'gamma': search_range}

grid_search = GridSearchCV(SVC(), svc_params, cv=5, return_train_score=True)
grid_search.fit(X_train, y_train)

print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))