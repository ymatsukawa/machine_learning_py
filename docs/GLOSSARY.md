# Related Perceptron

## Optimization problem

Finding best "workable" or "recommended" solution.

* https://community.fico.com/s/what-is-mathematical-optimization

## Objective function

A function need to be minimized or maximized.

This takes "data" and "model parameters" as arguments.

* http://kronosapiens.github.io/blog/2017/03/28/objective-functions-in-machine-learning.html
* https://stats.stackexchange.com/questions/179026/objective-function-cost-function-loss-function-are-they-the-same-thing

## Cost function

Finding set of weights and biases that minimize the cost.

Cost means "how wrong the model is in terms of its ability to estimate the relationship between input and output data"

* https://towardsdatascience.com/machine-learning-fundamentals-via-linear-regression-41a5d11f5220

## Grdient descending

Optimization algorithm minimizing object(function) iteratively by updating parameter.

* https://ruder.io/optimizing-gradient-descent/
* https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html
* https://medium.com/swlh/gradient-descent-algorithm-3d3ba3823fd4

# Related logistic regression

## Odds ratio

Measure of how strongly an event is associated with two factors.

```
f(p) = p / (1 - p)

0 <= p < 1
p is stochastic variable
```

* https://www.ncbi.nlm.nih.gov/books/NBK431098/
* https://medium.com/@analyttica/odds-ratio-a8315f159307

## Logit function

inverse of odds ratio function.

```
f(p) = log(p / (1 - p))

0 <= p < 1
p is stochastic variable
```

* https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/logoddra.htm
* https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/logoddra.htm

## Sigmoid function

Returns value between 0 and 1. Useful to binary classification.

```
f(x) = 1 / (1 + exp(-1))
```

For example, when determine image of animal is dog or not, use this.

* value is greater or equal to 0.51, may be it's dog.
* less than 0.51, it may not be dog

* https://medium.com/@gabriel.mayers/sigmoid-function-explained-in-less-than-5-minutes-ca156eb3049a
* https://www.sciencedirect.com/topics/computer-science/sigmoid-function

## stratify

In order to decrease bias of test set, slice or divide more sample from test set.

* https://medium.com/@dhivyarao94/stratified-sampling-machine-learning-b622189ae77
* https://medium.com/analytics-vidhya/stratified-sampling-in-machine-learning-f5112b5b9cfe