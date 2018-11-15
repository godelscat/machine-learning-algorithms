import numpy as np
import pandas as pd
import cv2
import sys
import sys.path.append("../perceptron/")

from perceptron import train_split

class AdaBoost:

    # image thresholding
    @staticmethod
    def binarization(raw_data):
        data = np.zeros(raw_data.shape)
        row = data.shape[0]
        for i in range(row):
            cv_img = raw_data[i].astype(np.uint8)
            cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)
            data[i] = cv_img
        return data

    def __init__(self, M=10):
        self.M = M
        self.w = None # weights
        self.alpha = [] # coeffients of base function 
        self.base = [] # list of base function 
        self.axis = None

    # gini index Gini(D)
    def _gini(self, labels):
        d_size = labels.size
        val = set(labels)
        times = []
        for i in val:
            times.append(labels[labels == i].size)
        times = np.array(times)
        times = ( times /  d_size ) ** 2
        g = 1 - np.sum(times)
        return g
    
    # conditional gini index Gini(D,A)
    def _cond_gini(self, d_size, d1_labels, d2_labels):
        g_d1 = self._gini(d1_labels)
        g_d2 = self._gini(d2_labels)
        g = (d1_labels.size / d_size) * g_d1 + (d2_labels.size / d_size) * g_d2
        return g

    # choose the labels
    def _max_label(self, labels):
        a = np.sum(labels)
        la = 1 if a >= 0 else -1
        return la

    # pick the feature and split value
    def _feature(self, data, labels):
        nsamples = labels.size
        predict = np.zeros((nsamples))
        min_gini = np.Inf  # gini <= 1
        ax = None  
        ax_val = None 
        for i in self.axis:
            ax_data = data[:,i]
            val = set(ax_data)
            for j in val:
                idx_1 = np.argwhere(ax_data == j).flatten()
                idx_2 = np.argwhere(ax_data != j).flatten()
                d1_labels = labels[idx_1]
                d2_labels = labels[idx_2]
                d_size = labels.size
                temp_gini = self._cond_gini(d_size, d1_labels, d2_labels)
                if temp_gini < min_gini:
                    min_gini = temp_gini
                    ax = i
                    ax_val = j
                    la1 = self._max_label(d1_labels)
                    la2 = self._max_label(d2_labels)
                    predict[idx_1] = la1
                    predict[idx_2] = la2
        return ax, ax_val, la1, la2, predict

    # cal Zm Eq(8.5)
    def _cal_Z(self, predict, labels, m):
        exp_product = np.exp(-1.0 * self.alpha[m] * labels * predict)
        product = self.w * exp_product
        zm = np.sum(product)
        return zm

    # cal em Eq(8.1)
    def _cal_E(self, predict, labels, m):
        not_eq_idx = np.argwhere(predict != labels).flatten()
        em = np.sum(self.w[not_eq_idx])
        return em

    def train(self, data, labels):
        data = self.binarization(data)
        nsamples = labels.size
        ndim = data.shape[1]
        self.w = np.zeros((nsamples))
        self.axis = np.arange(ndim)
        for m in range(self.M):
            print("step : {}".format(m))
            ax, ax_val, la1, la2, predict = self._feature(data, labels)
            self.base.append((ax, ax_val, la1, la2))
            em = self._cal_E(predict, labels, m)
            alpham = 0.5 * np.log((1-em) / em)
            self.alpha.append(alpham)
            zm = self._cal_Z(predict, labels, m)
            self.w = self.w * np.exp(-1.0 * alpham * labels * predict) / zm

    def predict(self, test, labels):
        test = self.binarization(test)
        nsamples = labels.size
        count = 0
        for i in range(nsamples):
            result = 0
            for m in range(self.M):
                ax = self.base[m][0]
                ax_val = self.base[m][1]
                if test[i, ax] == ax_val:
                    p = self.base[m][2]
                else:
                    p = self.base[m][3]
                result += self.alpha[m] * p
            predict = 1 if result >=0 else -1
            if predict == labels[i]:
                count += 1
        accuracy = format(count / nsamples, '.5f')
        print("Model accuracy : {}".format(accuracy))

if __name__ == "__main__":

    raw_data = pd.read_csv("../dataset/MNIST/binary_train.csv")
    labels = raw_data['label'].values
    data = raw_data.iloc[:,1:].values
    train_data, train_labels, test_data, test_labels = train_split(data, labels)

    model = AdaBoost()
    print("start training")
    model.train(train_data, train_labels)
    print("training complete")
    print("start predicting")
    model.predict(test_data, test_labels)
