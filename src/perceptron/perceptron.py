import numpy as np

class Perceptron:
    """
    ATTRIBUTES:
    
    eta: float
        learning rate (between 0.0 and 1.0)
        it adjusts weights with respect to loss gradient
        ref.
            https://heartbeat.fritz.ai/introduction-to-learning-rates-in-machine-learning-6ed685c16506
        learning rate real life
            If a child sees ten birds and all of them are black in color,
            he might believe that all birds are black and would consider this as a feature when trying to identify birds.
            Imagine next he’s shown a yellow bird, and his parents tell him that it’s a bird.
            With a desirable learning rate, he would quickly understand that black color is not an important feature of birds and would look for another feature.
            But with a low learning rate, he would consider the yellow bird an outlier and would continue to believe that all birds are black.
            And if the learning rate is too high, he would instantly start to believe that all birds are yellow even though he has seen more black birds than yellow ones.
            - Reprint source: https://heartbeat.fritz.ai/introduction-to-learning-rates-in-machine-learning-6ed685c16506
        finding good learning rate:
                https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10

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
        _errors: list
            number of misclassifications updates in each epoch
    
    Reference:
        https://www.quora.com/What-does-fitting-a-model-in-machine-learning-mean
        https://www.datarobot.com/wiki/fitting/
        https://en.wikipedia.org/wiki/Overfitting
    """
    def fit(self, X, y):
        # https://numpy.org/doc/1.13/reference/generated/numpy.random.RandomState.normal.html#numpy.random.RandomState.normal
        rand_generator = np.random.RandomState(self.random_state)
        self._w = rand_generator.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        
        self._errors = []
        for _ in range(self.data_num):
            errors = 0
            for xi, target in zip(X, y):
                """
                update weights [w1, w2, ... wn]
                ref.
                    https://lp-tech.net/articles/QUrxD
                    https://laptrinhx.com/supervised-learning-neural-networks-714798990/#pid=9
                """
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
        return np.dot(X, self._w[1:]) + self._w[0]
        
    
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
        return np.where(self.net_input(X) >= 0.0, 1, -1)
