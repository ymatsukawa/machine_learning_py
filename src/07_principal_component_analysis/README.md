# Principal Component Analysis

* (1) standardize data
* (2) create covariance-matrix from above data
* (3) get eigenvalue and eigenvecor from the matrix
* (4) by descending sort eigenvalue, rank engenvecor

## (1) standardize data

In statistics, standardization means "convert data X as mean is 0 and variance is 1".

When "mean" presents `x` and "standard deviation" shows `s`, standardization goes to

```
(X - x) / s
```