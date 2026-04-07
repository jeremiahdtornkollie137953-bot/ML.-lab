import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)


y = pd.Series(data.target, name="Target")

X.head()

X.isnull().sum()

X[:5]

#Train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)

#pipline
pipepline=Pipeline([('scaler',StandardScaler()),('svm',SVC())])


#LINEAR KERNEL

param_grid_linear = {
    'svm__kernel': ['linear'],
    'svm__C': [0.1, 1, 10]
}

grid_linear = GridSearchCV(
    pipepline,
    param_grid_linear,
    cv=5,
    scoring='accuracy'
)

grid_linear.fit(X_train, y_train)

print("LINEAR Kernel Best Params:", grid_linear.best_params_)
print("LINEAR Kernel Accuracy:", grid_linear.best_score_)


#POLYNOMIAL KERNEL

param_grid_poly = {
    'svm__kernel': ['poly'],
    'svm__C': [0.1, 1, 10],
    'svm__degree': [2, 3]
}

grid_poly = GridSearchCV(
    pipepline,
    param_grid_poly,
    cv=5,
    scoring='accuracy'
)

grid_poly.fit(X_train, y_train)

print("POLY Kernel Best Params:", grid_poly.best_params_)
print("POLY Kernel Accuracy:", grid_poly.best_score_)


#RBFKERNEL

param_grid_rbf = {
    'svm__kernel': ['rbf'],
    'svm__C': [0.1,1,10],
    'svm__gamma':[0.1,1,10]
}

grid_rbf = GridSearchCV(
    pipepline,
    param_grid_rbf,
    cv=5,
    scoring='accuracy'
)
grid_rbf.fit(X_train, y_train)

print('RBF Kernel BEst Params:',grid_rbf.best_params_)
print('RBF Kernel Accuracy:',grid_rbf.best_score_)


#SIGMOID KERNEL

param_grid_sigmoid = {
    'svm__kernel': ['sigmoid'],
    'svm__C': [0.1,1,10],
    'svm__gamma':[0.1,1,10]
}

grid_sigmoid = GridSearchCV(
    pipepline,
    param_grid_sigmoid,
    cv=5,
    scoring='accuracy'
)
grid_sigmoid.fit(X_train, y_train)


print('SIGMOID Kernel BEst Params:',grid_sigmoid.best_params_)
print('SIGMOID Kernel Accuracy:',grid_sigmoid.best_score_)


#CV=10 KERNEL

param_grid_linear = {
    'svm__kernel': ['linear'],
    'svm__C': [0.1, 1, 10]
}

grid_linear = GridSearchCV(
    pipepline,
    param_grid_linear,
    cv=10,
    scoring='accuracy'
)

grid_linear.fit(X_train, y_train)

print("LINEAR Kernel Best Params:", grid_linear.best_params_)
print("LINEAR Kernel Accuracy:", grid_linear.best_score_)


#POLYNOMIAL KERNEL CV=10

param_grid_poly = {
    'svm__kernel': ['poly'],
    'svm__C': [0.1, 1, 10],
    'svm__degree': [2, 3]
}

grid_poly = GridSearchCV(
    pipepline,
    param_grid_poly,
    cv=10,
    scoring='accuracy'
)

grid_poly.fit(X_train, y_train)

print("POLY Kernel Best Params:", grid_poly.best_params_)
print("POLY Kernel Accuracy:", grid_poly.best_score_)


#RBF KERNEL CV=10

param_grid_rbf = {
    'svm__kernel': ['rbf'],
    'svm__C': [0.1,1,10],
    'svm__gamma':[0.1,1,10]
}

grid_rbf = GridSearchCV(
    pipepline,
    param_grid_rbf,
    cv=10,
    scoring='accuracy'
)
grid_rbf.fit(X_train, y_train)

print('RBF Ki BEst Params:',grid_rbf.best_params_)
print('RBF Kernel Accuracy:',grid_rbf.best_score_)


#SIGMOID KERNEL CV=10

param_grid_sigmoid = {
    'svm__kernel': ['sigmoid'],
    'svm__C': [0.1,1,10],
    'svm__gamma':[0.1,1,10]
}

grid_sigmoid = GridSearchCV(
    pipepline,
    param_grid_sigmoid,
    cv=10,
    scoring='accuracy'
)
grid_sigmoid.fit(X_train, y_train)


print('SIGMOID Kernel BEst Params:',grid_sigmoid.best_params_)
print('SIGMOID Kernel Accuracy:',grid_sigmoid.best_score_)


#KNN USIN GRID SEARCH

from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier()

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

print("Best Accuracy:", grid.best_score_)
