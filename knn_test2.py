import numpy as np
import pandas as pd
from KNearestRegressor import KNearestRegressors

#taking  inputs
data=pd.read_csv('Social_Network_Ads.csv')
X=data.iloc[:,2:4].values
y=data.iloc[:,-1].values
#using train_test_split function
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#using StandardScaler function
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#an object of knn
knn=KNearestRegressors(k=5)
knn.fit(X_train,y_train)

#using the predict() function
knn.predict(np.array(X_test).reshape(len(X_test), len(X_test[0])))








