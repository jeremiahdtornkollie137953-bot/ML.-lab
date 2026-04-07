import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

#Breast Cancer dataset
data=load_breast_cancer()
X=data.data

#0=malignat, 1=benign

Y=data.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=42, stratify=Y)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, Y_train)

#predication and confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(Y_test,y_pred)

print("Confusion Matrix \n", cm)
tn, fp, fn, tp = cm.ravel()

print(f'True Positives (tp):{tp}')
print(f'True Negatives (tn):{tn}')
print(f'False Positives (fp):{fp}')
print(f'False Positives (fn):{fn}')

#Another way to extract tn,fp,fn,tp
TN=cm[0, 0]
FP=cm[0, 1]
FN=cm[1, 0]
TP=cm[1, 1]

print("\n Extracted_vallues:")
print("Tp:",TP)
print("Tn:",TN)
print("Fp:",FP)
print("Fn:",FN)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#accuracy
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)

#precision
precision = precision_score(Y_test,y_pred)
print("Precision:", precision)

#recall
recall=recall_score(Y_test,y_pred)
print("Recall:", recall)

#f1 score
f1 = f1_score(Y_test, y_pred)
print("F1 Score: ", f1)


#MANUAL CALCULATION USING CONFUSION MATRIX

#accuracy
accuracy=(tp+tn)/(tp+tn+fp+fn)
print("Accuracy:", accuracy)

#precision
precision=tp/(tp+fn)
print("Precision:", precision)

#recall
recall=tp/(tp+fn)
print("Recall:", recall)

#f1 score
f1=2*(precision*recall)/(precision+recall)
print("F1 Score:", f1)

#specificity
specificity = tn/(tn+fp)
print("Specificity:", specificity)



