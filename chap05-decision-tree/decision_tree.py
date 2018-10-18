# Here we use C4.5 algorithm with pruning

import numpy as np
import pandas as pd
import cv2
import sys
sys.path.append('../chap02-perceptron/')

from perceptron import train_split

# code in above ref, thresholding
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
        self.children = {} # dict of ax val and child node
        self.parent = None
        self.ax = None # decision feature on the node

class DTree():

    def __init__(self, epsilon=0.3):
        self.root = Node()
        self.epsilon = epsilon
        self.leaf = [] # collections of leaf nodes

    # empirical entropy H(D)
    def _emp_entropy(self, labels):
        d_size = labels.size
        val = set(labels)
        times = []
        for i in val:
            times.append(labels[labels == i].size)
        times = np.array(times)
        temp = np.log2(times / d_size)        
        temp[temp == np.NINF] = 0
        temp = - temp * (times / d_size)
        ent = np.sum(temp)
        return ent

    # empirical conditional entropy H(D|A)
    def _cond_entropy(self, ax, data, labels):
        d_size = labels.size
        ax_data = data[:,ax]
        ax_val = set(ax_data)
        d_times = [] # |D_i|
        for i in ax_val:
            d_times.append(ax_data[ax_data == i].size)
        d_times = np.array(d_times) 

        d_ent = []
        for i in ax_val:
            idx = np.argwhere(ax_data == i).flatten()
            temp = labels[idx]
            temp_ent = self._emp_entropy(temp)
            d_ent.append(temp_ent)
        d_ent = np.array(d_ent)
        ent = np.sum(d_ent * d_times) / d_size  # H(D|A)

       # feature A entropy H_A(D) 
        temp = np.log2(d_times / d_size)
        temp[temp == np.NINF] = 0
        temp = temp * (d_times / d_size)
        ax_ent = - np.sum(temp)

        return ax_ent, ent

    # information gain g(D|A)
    def _info_gain(self, ax, data, labels):
        hd = self._emp_entropy(labels)
        _, hda = self._cond_entropy(ax, data, labels)
        info = hd - hda
        return info

    # information gain ratio g_A(D|A)
    def _info_gain_ratio(self, ax, data, labels):
        hd = self._emp_entropy(labels)
        had, hda = self._cond_entropy(ax, data, labels)
        # H_A(D) could be zero
        assert had != 0
        info = (hd - hda) / had
        return info

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
            self.leaf.append(root_node)
            return 
            
        if not axises:
            root_node.label = self._max_label(labels)
            self.leaf.append(root_node)
            return 

        info_gain_list = []
        for ax in axises:
            info_gain = self._info_gain(ax, data, labels)
            info_gain_list.append(info_gain)
        
        if max(info_gain_list) < self.epsilon:
            root_node.label = self._max_label(labels)
            self.leaf.append(root_node)
            return
        
        idx = info_gain_list.index(max(info_gain_list))
        ag = axises.pop(idx)
        root_node.ax = ag
        ax_data = data[:,ag]
        ax_val = set(ax_data)
        for ax in ax_val:
            temp_idx = np.argwhere(ax_data == ax).flatten()
            child_node = Node()
            child_node.parent = root_node
            root_node.children[ax]  = child_node
            child_labels = labels[temp_idx]
            child_data = data[temp_idx]
           # child_node.label = self._max_label(child_labels)
            self._build(child_node, child_data, child_labels, axises)

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
            while temp_node.children:
                ax = temp_node.ax
                val = test[i,ax] 
                assert val in temp_node.children
                temp_node = temp_node.children[val]
            if temp_node.label == labels[i]:
                counts += 1
        accuracy = format(counts / nevents, '.5f')
        print("Model accuracy is:{}".format(accuracy))
                

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
