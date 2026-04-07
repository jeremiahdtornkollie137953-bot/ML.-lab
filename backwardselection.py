import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv(r"C:\Users\Dell X360\OneDrive\Desktop\ML\jere proj.csv")

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

# Convert ALL columns to numeric where possible
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop missing values
df = df.dropna()

# Drop non-numeric columns (Time is not useful for prediction)
if "Time" in df.columns:
    df = df.drop(columns=["Time"])

# 2. Define Features and Target
X = df.drop(columns=["Class"])
Y = df["Class"]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 3. Backward Feature Selection
selected_features = list(X.columns)

model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
model.fit(X_train[selected_features], Y_train)

y_pred = model.predict(X_test[selected_features])
best_score = accuracy_score(Y_test, y_pred)

print("Backward feature elimination process:\n")
print(f'Initial Accuracy (All features): {best_score:.4f}\n')

while len(selected_features) > 1:
    scores = []

    for feature in selected_features:
        features_to_test = selected_features.copy()
        features_to_test.remove(feature)

        model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
        model.fit(X_train[features_to_test], Y_train)

        y_pred = model.predict(X_test[features_to_test])
        score = accuracy_score(Y_test, y_pred)

        scores.append((score, feature))

    # Sort by best score
    scores.sort(reverse=True)
    current_best_score, worst_feature = scores[0]

    if current_best_score >= best_score:
        best_score = current_best_score
        selected_features.remove(worst_feature)

        print(f"Removed: {worst_feature}, Accuracy: {best_score:.4f}")
    else:
        break

print("\nFinal selected Features:")
print(selected_features)