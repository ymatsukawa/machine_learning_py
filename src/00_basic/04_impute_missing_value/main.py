from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import os.path as p
current_dir = p.dirname(p.realpath(__file__))

df = pd.read_csv(p.join(current_dir, './missing.csv'))

# all dataframe
print(df)

# drop columns where includs Nan(= None)
print(df.dropna())

# mean impute to missing value
imputer = SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None)
imputer.fit(df.values)
imputed = imputer.transform(df.values)
print(imputed)