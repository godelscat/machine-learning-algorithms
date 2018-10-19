# CART algorithm : classification tree
# final accuracy is about 20%, too low!

import numpy as np
import pandas as pd 
import cv2
import sys
sys.path.append("../perceptron/")

from perceptron import train_split


def binarization(raw_data):
    data = np.zeros(raw_data.shape)
    row = data.shape[0]
    for i in range(row):
        cv_img = raw_data[i].astype(np.uint8)
        cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)
        data[i] = cv_img
    return data

class Node():
    
    def __init__(self):
        self.label = None
        self.left = None
        self.right = None
        self.parent = None
        self.ax = None
        self.ax_val = None

class DTree():

    def __init__(self, epsilon=0.1, limits=100):
        self.root = Node()
        self.epsilon = epsilon
        self.limits = limits

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

    def _max_label(self, labels):
        val = list(set(labels))
        times = []
        for i in val:
            temp = labels[labels == i].size
            times.append(temp)
        idx = times.index(max(times))
        return val[idx]

    def _build(self, root_node, data, labels, axises):

        label_val = list(set(labels))
        if len(label_val) == 1: 
            root_node.label = label_val[0]
            return 

        if not axises:
            root_node.label = self._max_label(labels)
            return
    
        """
        if labels.size < self.limits:
            root_node.label = self._max_label(labels)
            return 
        """

        """
        g_data = self._gini(labels)
        if g_data < self.epsilon:
            root_node.label = self._max_label(labels)
            return 
        """        

        min_gini = 1  # gini <= 1
        ax = 0  
        ax_val = 0
        for i in axises:
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

        LNode = Node()
        RNode = Node()
        LNode.parent = root_node
        RNode.parent = root_node
        root_node.left = LNode
        root_node.right = RNode
        root_node.ax = ax
        root_node.ax_val = ax_val
        axises = axises.remove(ax)
        idx_1 = np.argwhere(data[:,ax] == ax_val).flatten()
        idx_2 = np.argwhere(data[:,ax] != ax_val).flatten()
        self._build(LNode, data[idx_1], labels[idx_1], axises) 
        self._build(RNode, data[idx_2], labels[idx_2], axises) 
        return

    def train(self, data, labels):
        
        ndim = data.shape[1]
        axises = list(range(ndim))
        self._build(self.root, data, labels, axises)

    def predict(self, test, labels):
    
        counts = 0
        nevents = labels.size
        for i in range(nevents):
            temp_node = self.root
            while (temp_node.left != None) or (temp_node.right != None):
                idx = temp_node.ax
                if test[i, idx] == temp_node.ax_val:
                    temp_node = temp_node.left
                else:
                    temp_node = temp_node.right
            if labels[i] == temp_node.label:
                counts += 1
        accuracy = format(counts / nevents, '.5f')
        print("Model accuracy is: {}".format(accuracy))


if __name__ == "__main__":

    raw_data = pd.read_csv("../dataset/MNIST/train.csv")
    labels = raw_data['label'].values
    data = raw_data.iloc[:,1:].values
    data = binarization(data)
    train_data, train_labels, test_data, test_labels = train_split(data, labels)

    model = DTree()
    print("start training")
    model.train(train_data, train_labels)
    print("training complete")
    print("start predicting")
    model.predict(test_data, test_labels)
