import os
from multiprocessing import Process
import numpy as np
import math
from collections import Counter
from pandas import read_csv


class Point:
    def __init__(self,data,cat):
        self.data=np.array(data)
        self.cat=cat

class KNNClassifier:

    k=5
    dmethod="M"
    def __init__(self,k=5,dmethod="M"):
        self.k=k
        self.dmethod=dmethod
        self.points=[]

    def fit(self,points):
        self.points=points
    def predict(self,point):
        neighbors=[ [None,math.inf] for _ in range(self.k)]
        for i in self.points:
            d=self.Distance(point,i.data)
            if (d<neighbors[0][1]):
                neighbors=neighbors[1:]
                neighbors.append([i,d])
        cats=Counter([t[0].cat for t in neighbors])
        return cats.most_common(1)[0][0]
            
    def Distance(self,x,y,method="M"):
        distance=0
        if (len(x)!=len(y)):
            raise Exception("Length of X and Y is not equal!")
        if (method=="M"):
            for i in range(len(x)):
                distance+=abs(y[i]-x[i])
            return distance
        else:
            for i in range(len(x)):
                distance+=pow(y[i]-x[i],2)
            return math.sqrt(distance)

def test():
    data=[[2,4],[1,3],[2,3],[3,2],[2,1],[5,6],[4,5],[4,6],[6,6],[5,4]]
    cat=["blue","blue","blue","blue","blue","red","red","red","red","red",]
    points=[Point(data[p],cat[p]) for p in range(len(data)) ]
    cl=KNNClassifier(k=5,dmethod="M")
    cl.fit(points)

    print(cl.predict([3,3]))
def read_from_file(filename):
        #data=read_csv(filename)
        #p=[d for d in data.loc[:,data.columns!="Class"]]
        #c=[d for d in data.loc[:,data.columns=="Class"]]

        f=open(filename,"r")
        Lines = f.readlines()
        Points=[]
        for line in Lines[1:]:
            s=line.split(",")
            Points.append(Point([int(x) for x in s[:len(s)-1]],s[-1]))
        return Points

        #return [Point(p[i],c[i]) for i in range(len(c))]
if __name__=='__main__':
    path=os.path.dirname(os.path.realpath(__file__))
    filename=path.replace("/src",'/data/KNNC.csv')


    cl=KNNClassifier(k=5,dmethod="E")
    cl.fit(read_from_file(filename))
    p=[]
    p.append(int(input("Give x")))
    p.append(int(input("Give y")))
    print(cl.predict(p))
        
