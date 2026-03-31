#PCA on auto pg dataset
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Load Dataset
df = pd.read_csv (r"C:\Users\Dell X360\OneDrive\Desktop\ML\jere proj.csv")

# Data processing
#REplaace ? with NAN
df.replace('?',np.nan, inplace=True)

#Drop rows with missing values
df.dropna(inplace=True)

#Use all numeric columns as features
x = df.copy()
y = None

#Feature scaling(Important for PCA)
scaler = StandardScaler()
x_scale = scaler.fit_transform(x)

#Apply PCA - use at most the number of features available
n_components = min(2, x_scale.shape[1])
pca = PCA(n_components=n_components)

x_pca = pca.fit_transform(x_scale)

#Results
print('Original shape', x_scale.shape)
print('Reduced shape', x_pca.shape)