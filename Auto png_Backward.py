import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv(r"C:\Users\Dell X360\OneDrive\Desktop\car data.csv")

# Drop Car_Name column
if 'Car_Name' in df.columns:
    df.drop(columns=['Car_Name'], inplace=True)

# Encode categorical columns
categorical_columns = ['Fuel_Type', 'Seller_Type', 'Transmission']
le_dict = {}
for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

X = df.drop(columns=['Selling_Price'])
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

selected_features = list(X.columns)
best_score = -np.inf

scores_progress = []
features_progress = []

while len(selected_features) > 1:
    scores = []

    for feature in selected_features:
        features_to_test = selected_features.copy()
        features_to_test.remove(feature)

        model = LinearRegression()

        cv_scores = cross_val_score(
            model,
            X_train_scaled[features_to_test],
            y_train,
            cv=5,
            scoring='r2'
        )

        mean_score = np.mean(cv_scores)
        scores.append((mean_score, feature))

    scores.sort(reverse=True)
    current_best_score, feature_to_remove = scores[0]

    if current_best_score > best_score:
        best_score = current_best_score
        selected_features.remove(feature_to_remove)

        scores_progress.append(best_score)
        features_progress.append(len(selected_features))
    else:
        break

final_model = LinearRegression()

final_model.fit(
    X_train_scaled[selected_features],
    y_train
)

y_pred = final_model.predict(
    X_test_scaled[selected_features]
)

final_r2 = r2_score(y_test, y_pred)

print("Selected Features:", selected_features)
print("Final R2 Score:", final_r2)

plt.figure()
plt.plot(features_progress, scores_progress, marker='o')
plt.xlabel("Number of Features")
plt.ylabel("Cross-Validated R2 Score")
plt.title("Backward Feature Selection Performance")

plt.savefig("backward_feature_selection.png")
plt.show()