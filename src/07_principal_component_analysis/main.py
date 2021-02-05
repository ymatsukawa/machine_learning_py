import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os.path as p
current_dir = p.dirname(p.realpath(__file__))

df_wine = pd.read_csv(p.join(current_dir, '../data/wine.csv'))

scaler = StandardScaler()
scaler.fit(df_wine.values)
X_scaled = scaler.transform(df_wine.values)

pca = PCA(n_components=2)
pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)
x = X_pca[:, 0]
y = X_pca[:, 1]

plt.figure(figsize=(8, 8))
plt.scatter(x, y, c=list(df_wine.iloc[:, 0]))
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig(p.join(current_dir, './pca.png'))