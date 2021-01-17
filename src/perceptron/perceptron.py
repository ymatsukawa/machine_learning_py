import numpy as np

class Perceptron:
    """
    ATTRIBUTES:
    
    eta: float
        learning rate (between 0.0 and 1.0)

    data_num: int
        training data set num
    
    random_state: int
        random generator seed for random weight
    """
    def __init__(self, eta, data_num, random_state):
        self.eta = eta
        self.data_num = data_num
        self.random_state = random_state
    
    """
    fit training data

    Parameters:
        X: {array-like}, shape = [n_samples, n_features]
            training vectors where
            number of samples and number of features.
        y: array-like, shape = [n_samples]
            target values.
    
    Returns:
        self : object

    Internal Attributes:
        _w: 1d-array
            weights after fitting
        _errors: list
            number of misclassifications updates in each epoch
    
    Reference:
        https://www.quora.com/What-does-fitting-a-model-in-machine-learning-mean
        https://www.datarobot.com/wiki/fitting/
        https://en.wikipedia.org/wiki/Overfitting
    """
    def fit(self, X, y):
        # https://numpy.org/doc/1.13/reference/generated/numpy.random.RandomState.normal.html#numpy.random.RandomState.normal
        rand_generator = np.random.RandomStae(self.random_state)
        self._w = rand_generator.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        
        self._errors = []
        for _ in range(self.data_num):
            errors = 0
            for xi, target in (X, y):
                update_w = self.eta * (target - self.predict(xi))
                self._w[1:] += update_w * xi
                self._w[0] += update_w
                errors += int(update_w != 0.0)
            self._errors.append(errors)
        
        return self
    
    """
    matrix product

    Paramteres:
        X: target matrix. X's dimension equals to self._w[1:]'s dimension

    Returns:
        value of Xi * wi + w{i-1}

    Reference:
        https://numpy.org/doc/1.13/reference/generated/numpy.dot.html#numpy.dot
    """
    def net_input(self, X):
        return np.dot(X, self._w[1:]) + self._w[0])
        
    
    """
    return class label after unit step
    
    Parameters:
        X: target matrix.
    
    Returns:
        1 or -1
        value is depends on net_input result.

    Reference:
        https://numpy.org/doc/stable/reference/generated/numpy.where.html
    """
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)