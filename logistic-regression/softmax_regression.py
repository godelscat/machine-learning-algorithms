# softmax regression : multi-nominal logistic regression
# ref: http://ufldl.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92
# in order to be stable in different kinds of kernal initializer, 
# I did some transform of equation p(y=j|x=xi)
import numpy as np
import pandas as pd
import sys
import time
from scipy.special import logsumexp
sys.path.append('../perceptron/')

from perceptron import train_split

class Softmax:

    def __init__(self, learning_rate=0.0001):
        self.learning_rate = learning_rate
        self.num_of_iter = 100000
        self.lam = 0.01
        self.num_of_class = 10 
        self.weights = None

    # probability of p(y=j|x=xi) = exp(wi*xi + bi) / norm
    def _probability(self, xi, j):
        # exp(wj*xi + bj) 
        # numerator = np.exp(np.dot(self.weights[j], xi))
        
        # denominator sum of exp(wj*xi + bj) over j
        # inner_product = np.sum(self.weights*xi, axis=1)
        # denominator = np.sum(np.exp(inner_product))
        # in order to avoid overflow, we need to do some transfrom of p(y=j|x=xi)
        reduce_weights = self.weights - self.weights[j]
        inner_product = np.sum(reduce_weights * xi, axis=1)
        # norm = np.exp(logsumexp(inner_product))
        # logsumexp costs a lot
        norm = np.sum(np.exp(inner_product))
        return  1 / norm 

    # gradient with respect to wj, randomly pick in xi direction
    # w is K by N array, b is K vector
    def _gradient(self, xi, yi, j):
        ins = 1 if yi == j else 0
        p = self._probability(xi, j)
        delta_wj = - xi * (ins - p) + self.lam * self.weights[j]
        return delta_wj

    def train(self, data, labels):
        nsamples = data.shape[0]
        ndim = data.shape[1]
        self.num_of_class = len(set(labels))
        # initialize weight and bias
        # it turns out combining weight and bias together works faster
        # You can use different initializer here
        self.weights = np.random.uniform(size=(self.num_of_class, ndim+1))

	# update
        for n in range(self.num_of_iter):
            idx = np.random.randint(0, nsamples) 
            xi = np.append(data[idx], 1.0)
            yi = labels[idx]
            for j in range(self.num_of_class):
                delta_wj = self._gradient(xi, yi, j)
                self.weights[j] -= self.learning_rate * delta_wj

    def predict(self, test, labels):
        nsamples = test.shape[0]
        ndim = test.shape[1]
        counts = 0
        for i in range(nsamples):
            p_list = []
            xi = np.append(test[i], 1.0)
            for j in range(self.num_of_class):
                p = self._probability(xi, j)
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
    start = time.time()
    model.train(train_data, train_labels)
    end = time.time()
    print("training cost : {}s".format(end - start))
    print("start predicting")
    model.predict(test_data, test_labels)
