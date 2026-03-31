import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#Load Dataset
df = pd.read_csv(r"C:\Users\Dell X360\OneDrive\Desktop\ML\jere proj.csv")  # Update path to your CSV file location
print(df.head())
df.replace('?', np.nan, inplace=True)
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df = df.dropna()

#Drop non-numeric (car name)

if 'car name' in df.columns:
    df = df.drop('car name', axis=1)

# Define features and target variable
x = df.drop(columns = ['mpg'])
y = df['mpg']

#Train - test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#remaining_features → features not yet selected
#selected_features → features already selected
#best_score → best R² score achieved so far
#We initialize best_score = -∞ so that the first feature always improves the score.

remaining_features = list(x.columns)
selected_features = []
best_score = -np.inf

print("Forward Feature Selection Process:\n")

#The loop continues until:
#No features remain, OR
#Adding a new feature does not improve performance

while remaining_features:
    scores = []
    # We temporarily add one feature at a time to the selected set. 
    for features in remaining_features:
        features_to_test = selected_features + [features]

        #Train regression model using only the selected + candidate feature.
        #This is the wrapper mechanism (model-based evaluation).

        model = LinearRegression()
        model.fit(X_train[features_to_test], y_train)

        y_pred = model.predict(X_test[features_to_test])
        score = r2_score(y_test, y_pred)

        scores.append((score, features))

    #Identify the feature that gives the best R² score when added to the selected set.
    scores.sort(reverse=True)
    current_best_score, best_feature = scores[0]

    #If adding the feature improves R²:
#Update best_score
#Add feature permanently
#Remove from remaining list
#Else:
#Stop algorithm (no further improvement)
    if current_best_score > best_score:
        best_score = current_best_score
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        print(f"Added: {best_feature}, R2 Score: {best_score:.4f}")
    else:
        break

print("\nSelected Features:", selected_features)