import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#sample dataset
data = load_breast_cancer()
x = data.data
y = data.target #0 = malignant, 1 = benign

#train test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)

#train model
model = LogisticRegression(max_iter=5000)
model.fit(x_train, y_train)

# Predictions and Confusion Matrix
y_predict = model.predict(x_test)
cm = confusion_matrix(y_test, y_predict)

print("Confusion Matrix:\n", cm)

tn, fp, fn, tp = cm.ravel()
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}") 
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

#Another way to extract tn, fp, fn, tp
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]

print(f"\n Extracted Values: TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")