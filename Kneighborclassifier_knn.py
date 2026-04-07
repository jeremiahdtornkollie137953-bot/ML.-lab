import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv(r"C:\Users\Dell X360\OneDrive\Desktop\ML\jere proj.csv")

print("Dataset Preview:")
print(df.head())

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

# Convert all columns to numeric where possible
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop missing values
df = df.dropna()

# Drop non-numeric column (Time is not useful for prediction)
if "Time" in df.columns:
    df = df.drop(columns=["Time"])
    
print("\nCleaned Dataset:")
print(df.head())

# All Columns Except Last (Class)
X = df.drop(columns=["Class"])

# Last Column (Class)
Y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Test k values from 1 to 20 (reasonable range)
max_k = min(20, len(X_train))

best_accuracy = 0
best_k = 1

for k in range(1, max_k + 1):

    print(f"\nK = {k}")

    classifier = KNeighborsClassifier(n_neighbors=k)

    # Train model
    classifier.fit(X_train, y_train)

    # Prediction
    y_pred = classifier.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Accuracy
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"Accuracy = {acc:.2f}%")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_k = k

print(f"\n\nBest K = {best_k} with Accuracy = {best_accuracy:.2f}%")
