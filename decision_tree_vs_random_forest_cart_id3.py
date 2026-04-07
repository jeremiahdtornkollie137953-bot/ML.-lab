import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = pd.read_csv(r"C:\Users\Dell X360\OneDrive\Desktop\ML\jere proj.csv")


X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#DECISION TREE USING GINI

dt_gini = DecisionTreeClassifier(criterion="gini", random_state=42)

dt_gini.fit(X_train, y_train)

y_pred_gini = dt_gini.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_gini))

print("\nConfusion Matrix:", confusion_matrix(y_test, y_pred_gini))

print("\nClassification Report:", classification_report(y_test, y_pred_gini))


#DECISION TREE USING ID3 (ENTROPY)

dt_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42)

dt_entropy.fit(X_train, y_train)
y_pred_entropy = dt_entropy.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_entropy))

print("\nConfusion Matrix:", confusion_matrix(y_test, y_pred_entropy))

print("\nClassification Report:", classification_report(y_test, y_pred_entropy))


#RANDOM FOREST CLASSIFIER

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_rf))

print("\nConfusion Matrix:", confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:", classification_report(y_test, y_pred_rf))






