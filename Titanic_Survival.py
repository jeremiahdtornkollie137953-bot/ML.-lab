# Importing libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv(r"C:\Users\Dell X360\Downloads\Titanic-Dataset.csv")  

# Drop irrelevant columns
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Handle missing values
age_imputer = SimpleImputer(strategy='median')
emb_imputer = SimpleImputer(strategy='most_frequent')

df['Age'] = age_imputer.fit_transform(df[['Age']]).ravel()
df['Embarked'] = emb_imputer.fit_transform(df[['Embarked']]).ravel()

# Encode categorical variables
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

# Split data into features and target
X = df.drop('Survived', axis=1)
Y = df['Survived']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression Model
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, Y_train)

# Predictions
Y_pred = log_reg_model.predict(X_test)

# Evaluation
print("Logistic Regression Accuracy:", accuracy_score(Y_test, Y_pred))
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))

# SVM Model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, Y_train)

# Predictions
svm_pred = svm_model.predict(X_test)

# Evaluation
print("\nSVM Accuracy:", accuracy_score(Y_test, svm_pred))
print("\nSVM Classification Report:\n", classification_report(Y_test, svm_pred))

# Model Comparison
print("\n--- Model Comparison ---")
print("Logistic Regression Accuracy:", accuracy_score(Y_test, Y_pred))
print("SVM Accuracy:", accuracy_score(Y_test, svm_pred))