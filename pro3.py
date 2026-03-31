#Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

#Load Iris Dataset
data = load_iris()
x = data.data
y = data.target

#Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)

for k in range(1,6):
    #Initialize KNN Classifier
    classifier = KNeighborsClassifier(n_neighbors=k)

    #Fitting the model
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    accuracy = accuracy_score(y_test, y_pred)*100
    print('The Accuracy of our model is ' + str(round(accuracy, 2)) + '%')