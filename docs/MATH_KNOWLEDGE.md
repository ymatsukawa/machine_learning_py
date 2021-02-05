# average and mean

In statistics, average and "mean" are "arithmetic average".

(In this doc, mean is used as "arithmetic average")

```
(Sum of data number) / (Total quantity of data)

data = 1, 2, 3, 4, 5
average = (1 + 2 + 3 + 4 + 5) / 5 = 3
```

ref.

* https://www.mathsisfun.com/mean.html
* https://www.vedantu.com/maths/difference-between-mean-and-average
* http://mathcentral.uregina.ca/qq/database/qq.09.00/julie1.html

# deviation

Difference between data and mean. When deviation is big, it's not average.

```
deviation = X - m

X = data
m = mean
```

ref.

* https://www.mathsisfun.com/data/mean-deviation.html
* https://www.mathsisfun.com/definitions/deviation.html

# variance

It tells how measured data get distance from the mean value of the dataset.

```
r^2 = Sigma{ (X - m)^2 } / N

Sigma = Sum of
X = each value
m = mean
N = number of dataset
```

ref.

* https://www.easycalculation.com/maths-dictionary/variance.html
* https://www.mathsisfun.com/definitions/variance.html

# standard deviation

It measures how much the data is spread out.

Value is positive square root of variance.

```
r = sqrt{ Sigma{ (X - m)^2 } / N }
```

ref.

* https://www.mathsisfun.com/data/standard-deviation.html
* https://www.statisticshowto.com/probability-and-statistics/standard-deviation/

# covariance

It measures link or relation of two sets and shows value.

When the value is positive, they have high relationship.

When the value is negative, they have low relationship.

```
Total sum of product of x's and y's variance

Sxy = (1/n) * Sigma(1 to n) {(xi - xm) * (yi - ym)}

n = total num of data
xm, ym = mean of x, mean of y
```

ref.

* https://www.mathsisfun.com/data/correlation.html
* https://www.statisticshowto.com/covariance/
* https://towardsdatascience.com/covariance-and-correlation-321fdacab168

# variance-covariance matrix

matrix which indactes how data spread or summarized.

```
* If the entries in the column vector are random value

X = [X1, X2, .., Xn]^T

* each with finite variance and expected value
* covariance matrix Kxx is the matrix whose matrix (i, j) is the covariance

K_xi_xj = cov[Xi, Xj] = E([Xi - E[Xi])(Xj - E[Xj])
= E[(X - μx)(X - μx)^T] = E[XX^T] - μxμx^T

μx = E[X]
E is expected value (mean) of its argument
```

* https://datascienceplus.com/understanding-the-covariance-matrix/
* https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc541.htm
* https://jp.mathworks.com/matlabcentral/answers/309253-covariance-matrix-from-a-random-vector