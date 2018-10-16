import numpy as np
import pandas as pd 
from collections import Counter
import sys 
sys.path.append('../chap02-perceptron/')

from perceptron import train_split
from kdTree import KDTree

#sys.setrecursionlimit(10**6)

class KNN():
	
    def __init__(self, k=5):
        self.k = k
        self.tree = KDTree()
    
    def _max_label(self, k_list):
        temp = [x.label for x in k_list]
        max_occ = Counter(temp).most_common(1)
        max_label = max_occ[0][0]
        return max_label 
    
    def train(self, data, labels):
        assert data[:,0].size == labels.size
        self.tree.build(data, labels)	
            
    def predict(self, data, labels):
        assert data[:,0].size == labels.size
        nevents = labels.size
        counts = 0
        for i in range(nevents):
            k_list = []
            k_list = self.tree.kNN_points(data[i], self.k)	
            la = self._max_label(k_list)
            if la == labels[i]:
                counts += 1
                print(counts)
        accuracy = format(counts/nevents, '.5f')
        print("kNN Model accuracy is : {}".format(accuracy))


if __name__ == "__main__":

    raw_data = pd.read_csv("../dataset/MNIST/train.csv")
    labels = raw_data['label'].values
    data = raw_data.iloc[:,1:].values
    train_data, train_labels, test_data, test_labels = train_split(data, labels)

    print("start building model")
    model = KNN(k=10)
    print("start training")
    model.train(train_data, train_labels)
    print("start predicting")
    model.predict(test_data, test_labels)
    print("predicting ends")
    # final accuracy is around 95% 
    # But why is this kd-tree so slow
    # Even slower than naive search O(N^2)
