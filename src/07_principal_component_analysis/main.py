import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os.path as p
current_dir = p.dirname(p.realpath(__file__))

df_wine = pd.read_csv(p.join(current_dir, '../data/wine.csv'))
# TBD