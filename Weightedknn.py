import operator
from collections import Counter

import numpy as np


class KNearestNeighbors:
    def __init__(self,k,weights):
        self.k=k
        self.result=[]
        self.result_new=[]
        self.weights=weights


    def fit(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train
        print("Training Done")

    def predict(self,X_test):
        for j in X_test:
         distance={}
         new_dist={}
         counter=1
         for i in self.X_train:
              c=0
              for k in range(len(j)):
                  c=c+(j[k]-i[k])**2
              if self.weights == "uniform":
                  distance[counter] = c** 1 / 2
              elif self.weights == "distance":
                  new_dist[counter] = 1 /(c ** 1 / 2)
                  counter=counter+1
              if self.weights == "uniform":
                  distance = sorted(distance.items(), key=operator.itemgetter(1))
                  self.result.append(self.classify(distance[:self.k]))



              elif self.weights == "distance":
                  new_dist = sorted(new_dist.items(), key=operator.itemgetter(1), reverse=True)
                  self.result_new.append(self.classify_new(new_dist[:self.k]))


              if self.weights == "uniform":
                  return self.result
              elif self.weights == "distance":
                  return self.result_new
    def classify(self,distance):
        label=[]
        for i in distance:
            label.append(self.y_train[i[0]])

        return Counter(label).most_common()[0][0]
    def classify_new(self,distance):
        y = np.unique(self.y_train)
        [s]=len(y)
        for i in range(len(y)):
            sum = 0
            for j in distance:
                if y[i] == self.y_train[i[0]]:
                    sum=sum+i[1]
            s[i] = sum

        return y[max(s)]







