# imputation

## dataset

A | B | C | D
-- | -- | -- | --
1.0 | 2.0 | 3.0 | 4.0
5.0 | 6.0 | Nan | 8.0
9.0 | 10.0 | 11.0 | NaN

## drop without imputation

column with NaN is dropped

it's easy but data reliability will be lost.

A | B | C | D
-- | -- | -- | --
1.0 | 2.0 | 3.0 | 4.0

## impute with value

mean imputation to NaN data.

In this explain, "mean" says average of row.

A | B | C | D
-- | -- | -- | --
1.0 | 2.0 | 3.0 | 4.0
5.0 | 6.0 | **7.0** | 8.0
9.0 | 10.0 | 11.0 | **6.0**