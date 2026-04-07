from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#load the Iris Dataset
iris=load_iris()
X=iris.data
Y=iris.target

#split the dataset into training and testing sets 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

#create a decision tree classifier
clf=DecisionTreeClassifier(random_state=42)

#fit the classifier to the training data
clf.fit(X_train, Y_train)

#get feature importance
feature_importances=clf.feature_importances_

#print feature importances
for feature, importance in zip(iris.feature_names, feature_importances):
    print(f"{feature}: {importance}")

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, Y_train)
feature_importances = rf_clf.feature_importances_

#Extra tree classifier

et_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_clf.fit(X_train, Y_train)
feature_importances = et_clf.feature_importances_
