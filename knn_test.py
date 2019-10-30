import numpy as np
import pandas as pd
from KNearestNeighbor import KNearestNeighbors

#taking  inputs
data=pd.read_csv('Social_Network_Ads.csv')
data['Gender'] = data['Gender'].replace({'Male': 0, 'Female': 1})
X=data.iloc[:,1:4].values
y=data.iloc[:,-1].values

print(X.shape)
print(y.shape)

#using train_test_split function
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#using StandardScaler function
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#an object of knn
knn=KNearestNeighbors(k=5)
knn.fit(X_train,y_train)

#using the predict() function
knn.predict(np.array(X_test).reshape(len(X_test), len(X_test[0])))

#defining a function to check the output
def predict_new():
    age=int(input("Enter the age"))
    salary=int(input("Enter the salary"))
    gender = int(input("Enter the gender,type '0' for Male or type '1' for female"))
    X_new=np.array([[age],[gender],[salary]]).reshape(1,3)
    X_new=scaler.transform(X_new)
    result=knn.predict(X_new)
    if result==0:
        print("Will not purchase")
    else:
        print("Will purchase")



predict_new()




