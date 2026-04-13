import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

df = pd.read_csv(r"emails.csv")

# Drop the 'Email No.' column (non-numeric identifier)
if 'Email No.' in df.columns:
    df.drop(columns=['Email No.'], inplace=True)

X = df.drop('Prediction', axis=1)
y = df['Prediction']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
nb_pred = nb_model.predict(X_test_scaled)
results['Naive Bayes'] = accuracy_score(y_test, nb_pred)
print("Naive Bayes Accuracy:", results['Naive Bayes'])

cart_model = DecisionTreeClassifier(criterion='gini', random_state=42)
cart_model.fit(X_train_scaled, y_train)
cart_pred = cart_model.predict(X_test_scaled)
results['Decision Tree (CART)'] = accuracy_score(y_test, cart_pred)
print("Decision Tree (CART) Accuracy:", results['Decision Tree (CART)'])

id3_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
id3_model.fit(X_train_scaled, y_train)
id3_pred = id3_model.predict(X_test_scaled)
results['Decision Tree (ID3)'] = accuracy_score(y_test, id3_pred)
print("Decision Tree (ID3) Accuracy:", results['Decision Tree (ID3)'])

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
results['Random Forest'] = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", results['Random Forest'])

svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
results['SVM'] = accuracy_score(y_test, svm_pred)
print("SVM Accuracy:", results['SVM'])

models = list(results.keys())
accuracies = list(results.values())

plt.figure()
plt.bar(models, accuracies)
plt.title("Model Comparison")
plt.ylabel("Accuracy")

for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()