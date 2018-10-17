# imag binarization ref: https://blog.csdn.net/wds2006sdo/article/details/51967839
import numpy as np 
import pandas as pd 
import sys
sys.path.append('../chap02-perceptron/')
import cv2

from perceptron import train_split

# code in above ref, thresholding
def binarization(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)
    return cv_img

class NaiveBayes():

    def __init__(self, X=2, Y=10, lam=1):
        self.lam = lam # lambda value
        self.nx = X # number of possible value for sample features
        self.ny = Y # number of possible value for labels
        self.cond_pr = None   # probability for P(Y=c_k)
        self.marg_pr = None   # probability for P(X=a|Y=c_k)
    
    def train(self, data, labels):
        nsamples = data[:,0].size
        ndim = data[0,:].size
        self.marg_pr = np.zeros(self.ny) 
        self.cond_pr = np.zeros((self.nx, ndim, self.ny)) 
        
        for i in range(nsamples):
            self.marg_pr[labels[i]] += 1
        self.marg_pr = (self.marg_pr  + self.lam) / (nsamples + self.ny * self.lam)

        for i in range(nsamples):
            img = binarization(data[i,:])
            for j in range(ndim):
                self.cond_pr[img[j], j, labels[i]] += 1

        self.cond_pr = self.cond_pr + self.lam
        
        for i in range(self.ny):
            self.cond_pr[:,:,i] = self.cond_pr[:,:,i] / (nsamples * self.marg_pr[i] + self.nx * self.lam)

    def predict(self, test, labels):
        nsamples = labels.size
        ndim = test[0,:].size
        counts = 0
        for i in range(nsamples):
            img = binarization(test[i,:])
            predict_labels = [1] * self.ny 
            for j in range(self.ny):
                for k in range(ndim):
                    predict_labels[j] = predict_labels[j] * self.cond_pr[img[k], k, j]
                predict_labels[j] = predict_labels[j] * self.marg_pr[j]
            p_label = predict_labels.index(max(predict_labels))
            if p_label == labels[i]:
                counts += 1

        accuracy = format(counts / nsamples, '.5f')
        print("Model accuracy is : {}".format(accuracy))


if __name__ == "__main__" :

    raw_data = pd.read_csv("../dataset/MNIST/train.csv")
    labels = raw_data['label'].values
    data = raw_data.iloc[:,1:].values
    train_data, train_labels, test_data, test_labels = train_split(data, labels)

    model =  NaiveBayes()
    print("start training")
    model.train(train_data, train_labels)
    print("start predicting")
    model.predict(test_data, test_labels)
