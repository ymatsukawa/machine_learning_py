import numpy as np

class LogisticRegression:
    """
    ATTRIBUTES:
    
    eta: float
        learning rate (between 0.0 and 1.0)

    data_num: int
        training data set num
    
    random_state: int
        random generator seed for random weight
    """
    def __init__(self, eta=0.01, data_num=50, random_state=1):
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
        _cost: list
            cost of logist regression
    """
    def fit(self, X, y):
        rand_generator = np.random.RandomState(self.random_state)
        print(X)
        self._w = rand_generator.normal(loc=0.0, scale=0.01, size=1 + X.T.shape[1])
        print(self._w)
        self._cost = []
        # decreased frequent update
        for _ in range(self.data_num):
            net_input = self.net_input(X)
            predict_result = self.predict(net_input)

            errors = (y - predict_result)
            self._w[1:] += self.eta * X.T.dot(errors)
            self._w[0] += self.eta * errors.sum()

            cost = -y.dot(np.log(predict_result)) - ((1 - y).dot(np.log(1 - predict_result)))
            self._cost.append(cost)
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
        return np.dot(X, self._w[1:]) + self._w[0]
    
    """
    calculate sigmoid function.
    value of z is constrained between -250 to 250

    Parmeters:
        z: array

    Reference:
        https://numpy.org/doc/stable/reference/generated/numpy.clip.html
    """
    def sigmoid(sef, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    """
    return class label after unit step
    
    Parameters:
        X: target matrix.
    
    Returns:
        1 or -1
        value is depended on `net_input` result.

    Reference:
        https://numpy.org/doc/stable/reference/generated/numpy.where.html
    """
    def predict(self, X):
        return np.where(self.sigmoid(self.net_input(X)) >= 0.5, 1, -1)
