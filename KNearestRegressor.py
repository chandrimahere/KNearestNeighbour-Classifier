import operator
from collections import Counter

class KNearestRegressors:
    def __init__(self,k):
        self.k=k
        self.result=[]


    def fit(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train
        print("Training Done")

    def predict(self,X_test):
        distance={}
        counter=1
        for i in self.X_train:
            distance[counter]=((X_test[0][0]-i[0])**2+(X_test[0][1]-i[1])**2)**1/2
            counter=counter+1
        distance=sorted(distance.items(),key=operator.itemgetter(1))
        print(distance)
