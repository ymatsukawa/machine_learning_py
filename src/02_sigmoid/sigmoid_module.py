import numpy as np

"""
  ARGUMENTS:
    p: float: stochastic variable
        0 <= p < 1
"""
def odds_ratio(p):
    return p / (1 - p)

"""
  ARGUMENTS:
    p: stochastic variable
        0 <= p < 1
"""
def logit(p):
    return np.log([odds_ratio(p)])

"""
    ARGUMENTS:
        p: stochastic variable
            0 <= p < 1
    
    inverse function of logit
"""
def sigmoid(p):
    return 1.0 / (1.0 + np.exp(-p))
