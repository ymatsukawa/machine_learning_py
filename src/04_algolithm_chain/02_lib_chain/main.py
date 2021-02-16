from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=16)

model_pipe = [
    ("scaler", MinMaxScaler()),
    ("svm", SVC())
]
pipe = Pipeline(model_pipe)
pipe.fit(X_train, y_train)
print("test score: {:.2f}".format(pipe.score(X_test, y_test)))