# softmax regression : multi-nominal logistic regression
# ref: http://ufldl.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92
import numpy as np
import pandas as pd
import sys
import math
sys.path.append('../perceptron/')

from perceptron import train_split

class Softmax:

    def __init__(self, learning_rate=0.0001, num_of_class=10):
        self.learning_rate = learning_rate
        self.num_of_iter = 100000
        self.weight_decay = 0.01
        self.num_of_class = num_of_class
        self.weights = None
        self.bias = None

    # probability of p(y=j|x=xi) = exp(wi*xi + bi) / norm
    def _probability(self, xi, j):
        # inner product of wi xi
        inner_product = np.sum(self.weights*xi, axis=1) + self.bias
	# calculate sum_{1,K} exp(w*xi)
        norm = np.sum(np.exp(inner_product))
        # calculate exp(wj*xi)
        numerator = np.exp(np.sum(self.weights[j]*xi) + self.bias[j])
        return numerator / norm
        

    # gradient with respect to wj, randomly pick in xi direction
    # w is K by N array, b is K vector
    def _gradient(self, xi, yi, j):
        ins = 1 if yi == j else 0
        p = self._probability(xi, j)
        delta_wj = - xi * (ins - p) + self.weight_decay * self.weights[j]
        delta_bj = - (ins - p) + self.weight_decay * self.bias[j] 
        return delta_wj, delta_bj

    def train(self, data, labels):
        nsamples = data.shape[0]
        ndim = data.shape[1]
        # initialize weight and bias
        self.weights = np.random.uniform(size=(self.num_of_class, ndim))
        self.bias = np.random.uniform(size=(self.num_of_class))

	# update
        for n in range(self.num_of_iter):
            idx = np.random.randint(0, nsamples) 
            for j in range(self.num_of_class):
                delta_wj, delta_bj = self._gradient(data[idx], labels[idx], j)
                self.weights[j] -= self.learning_rate * delta_wj
                self.bias[j] -= self.learning_rate * delta_bj

    def predict(self, test, labels):
        nsamples = test.shape[0]
        ndim = test.shape[1]
        counts = 0
        for i in range(nsamples):
            p_list = []
            for j in range(self.num_of_class):
                p = self._probability(test[i], j)
                p_list.append(p)
            idx = p_list.index(max(p_list))
            if idx == labels[i]:
                counts += 1

        accuracy = format(counts / nsamples, '.5f')
        print("Model accuracy is : {} ".format(accuracy))


if __name__ == "__main__" :
    
    raw_data = pd.read_csv("../dataset/MNIST/train.csv")
    labels = raw_data['label'].values
    data = raw_data.iloc[:,1:].values
    train_data, train_labels, test_data, test_labels = train_split(data, labels)

    model =  Softmax()
    print("start training")
    model.train(train_data, train_labels)
    print("start predicting")
    model.predict(test_data, test_labels)
