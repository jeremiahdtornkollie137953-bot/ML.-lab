from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score #FOR APPLAY CONFUSION_MATRIX AND ACCURACY_SCORE
from sklearn.model_selection import train_test_split #FOR SPLITING DATA INTO TRAIN-TEST
from sklearn.datasets import load_breast_cancer #IMPORT DATASET
     

#LOAD DATASET
data = load_breast_cancer()
X=data.data
Y=data.target
     

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
     

for k in range(1,8):
    # Instantiate learning model (k = 1 to 7)

    classifier=KNeighborsClassifier(n_neighbors=k)

    #fitting the model
    classifier.fit(X_train,Y_train)

    #predicting the results
    y_pred=classifier.predict(X_test)

     #confution matrix applay for cheking accuracy
    cm=confusion_matrix(Y_test,y_pred)
    print(cm)
    accuracy=accuracy_score(Y_test,y_pred)*100

    print("Accuracy of our model is equal "+str(round(accuracy,2))+" %.")
     
