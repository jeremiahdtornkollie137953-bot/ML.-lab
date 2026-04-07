import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load dataset
df = pd.read_csv(r"C:\Users\Dell X360\OneDrive\Desktop\ML\jere proj.csv")

# replace '?' with NAN
df.replace("?", np.nan, inplace=True)

# convert all columns to numeric where possible
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# drop missing values
df = df.dropna()

# drop non-numeric column (Time is not useful for prediction)
if "Time" in df.columns:
    df = df.drop(columns=["Time"])

# define feature and target
X = df.drop(columns=["Class"])
Y = df["Class"]

# train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# remaining_features -> features not yet selected
# selected_features -> features already selected
# best_score -> best R2 score achieved so far
# We intialize best_score = -infinite so that the first feature always imporeves the score.

remaining_features = list(X.columns)
selected_features = []
best_score = -np.inf

print("Forward Feature Selection Process:\n")

# the loop continues until:
# No features remain ,OR
# Adding a new feature does not improve performance

while remaining_features:
    scores = []

    # We temporarily add one feature at a time to the selected set.
    for feature in remaining_features:
        feature_to_test = selected_features + [feature]

        # train classification model using only the selected + candidate feature
        # this is the wrapper mechanism (model-based evaluation ).
        model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
        model.fit(X_train[feature_to_test], Y_train)

        y_pred = model.predict(X_test[feature_to_test])
        score = accuracy_score(Y_test, y_pred)

        scores.append((score, feature))

    # sort features by accuracy score
    # select feature giving highest improvement.
    scores.sort(reverse=True)
    current_best_score, best_feature = scores[0]

#if adding the feature improves accuracy:
#update best_score
#add feature permanently
#remove from remaining list
#else:
#stop algorithm (no further improvement)

    if current_best_score > best_score:
        best_score = current_best_score
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

        print(f'Added: {best_feature}, Accuracy: {best_score:.4f}')
    else:
        break

print('\n Final Selected features:')
print(selected_features)
