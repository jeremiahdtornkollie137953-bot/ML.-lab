import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"C:\Users\Dell X360\OneDrive\Desktop\ML\jere proj.csv")
print(df.head())

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Drop the Time column (not useful for prediction)
if "Time" in df.columns:
    df = df.drop(columns=['Time'])

# Define features and target
X = df.drop(columns=['Class'])
y = df['Class']

feat_labels = X.columns

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Get feature importances
feature_importances = clf.feature_importances_

# Print feature importances
print("\nFeature Importances:\n")
for feature, importance in zip(feat_labels, feature_importances):
    print(f"{feature}: {importance:.6f}")

# Print sorted feature importances
print("\nSorted Feature Importances:\n")
indices = np.argsort(feature_importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], feature_importances[indices[f]]))
