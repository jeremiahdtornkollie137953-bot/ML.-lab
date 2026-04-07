import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes  import GaussianNB,MultinomialNB, BernoulliNB, ComplementNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,Binarizer

data=load_iris()
X=data.data
y=data.target

df = pd.DataFrame(data.data, columns=data.feature_names)
     
print("Dataset Preview:")
print(df.head())
