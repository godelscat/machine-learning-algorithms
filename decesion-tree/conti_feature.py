"""
Ref: https://blog.csdn.net/u012328159/article/details/79396893
CART decision tree algorithm for continuous-value features.
W/O tree pruning.
"""

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

SMALL = 1e-06

class Node:

    def __init__(self):
        self.label = None 
        self.children = {} # dict of ax val and child node
        self.parent = None
        self.ax = None # decision feature on the node
        self.cont = None

class DTree():

    def __init__(self, epsilon=0.3, min_data_in_leaf=10):
        self.root = Node()
        self.epsilon = epsilon
        self.min_data_in_leaf = min_data_in_leaf
        self.cont = [] # index of continuous feature
        self.cat = [] # index of non-continuous feature
        self.feature_list = {}

    # get continuous feature axis
    def _conti_feature(self, data):
        nsample, ndim = data.shape
        axises = [
            ax for ax in range(ndim) if 
            'float' in str(data[:, ax].dtype)
        ]
        self.cont = axises
        self.cat = [ax for ax in range(ndim) if ax not in  axises]
    
    def _get_cont_feature_list(self, data):
        for ax in self.cont:
            sort_ax_data = np.sort(data[:, ax])
            temp = [np.median(sort_ax_data[i:i+2]) for i in range(len(data)-1)]
            self.feature_list[ax] = temp
        
    
    # empirical entropy H(D)
    def _emp_entropy(self, labels):
        d_size = labels.size
        val = set(labels)
        times = [labels[labels == i].size for i in val]
        times = np.array(times)
        temp = np.log2(times / d_size + SMALL)        
        #temp[temp == np.NINF] = 0
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
        temp = np.log2(d_times / d_size + SMALL)
        #temp[temp == np.NINF] = 0
        temp = temp * (d_times / d_size)
        ax_ent = - np.sum(temp)

        return ax_ent, ent

    # cat feature conditional feature
    # ax_val: val to split feature
    def _cond_entropy_cont(self, ax_val, ax_data, labels):
        d_size = labels.size
        d_times = []
        d_times.append(ax_data[ax_data <= ax_val].size)
        d_times.append(ax_data[ax_data > ax_val].size)
        d_times = np.asarray(d_times)
        
        d_ent = []
        idx_left = np.argwhere(ax_data <= ax_val).flatten()
        label_left = labels[idx_left]
        ent_left = self._emp_entropy(label_left)
        d_ent.append(ent_left)

        idx_right = np.argwhere(ax_data > ax_val).flatten()
        label_right = labels[idx_right]
        ent_right = self._emp_entropy(label_right)
        d_ent.append(ent_right)
    
        d_ent = np.asarray(d_ent)
        ent = np.sum(d_ent * d_times) / d_size

       # feature A entropy H_A(D) 
        temp = np.log2(d_times / d_size + SMALL)
        #temp[temp == np.NINF] = 0
        temp = temp * (d_times / d_size)
        ax_ent = - np.sum(temp)

        return ax_ent, ent

    # information gain g(D|A)
    def _info_gain(self, ax, data, labels, cat=True):
        hd = self._emp_entropy(labels)
        if cat: 
            _, hda = self._cond_entropy(ax, data, labels)
        else:
            _, hda = self._cond_entropy_cont(ax, data, labels)
        info = hd - hda
        return info

    # information gain ratio g_A(D|A)
    def _info_gain_ratio(self, ax, data, labels, cat=True):
        hd = self._emp_entropy(labels)
        if cat:
            had, hda = self._cond_entropy(ax, data, labels)
        else:
            had, hda = self._cond_entropy_cont(ax, data, labels)
        # H_A(D) could be zero
        #assert had != 0
        info = (hd - hda) / (had + SMALL)
        return info
    
    def _max_info_gain_cont(self, ax, data, labels):
        ax_val_list = self.feature_list[ax]
        info_list = []
        for ax_val in ax_val_list:
            temp = self._info_gain(ax_val, data[:, ax], labels, cat=False)
            info_list.append(temp)
        max_info = max(info_list) 
        max_idx = info_list.index(max_info)
        return max_info, max_idx
    
    def _max_label(self, labels):
        val = list(set(labels))
        times = [labels[labels == i].size for i in val]
        idx = times.index(max(times))
        return val[idx]

    def _build(self, root_node, data, labels, axises):
        label_val = list(set(labels))
        if len(label_val) == 1: 
            root_node.label = label_val[0]
          #  self.leaf.append(root_node)
            return 
        
        if labels.size < self.min_data_in_leaf:
            root_node.label = self._max_label(labels)
            return
            
        if not axises:
            root_node.label = self._max_label(labels)
            return 

        info_gain_list_cat = []
        for ax in self.cat:
            info_gain = self._info_gain(ax, data, labels)
            info_gain_list_cat.append(info_gain)
        
        cont_ax = None
        cont_gain = 0
        cont_idx = None
        for ax in self.cont:
            info_gain, info_gain_idx = self._max_info_gain_cont(ax, data, labels)
            if info_gain > cont_gain:
                cont_ax = ax
                cont_gain = info_gain
                cont_idx = info_gain_idx
        
        
        """
        if max(info_gain_list) < self.epsilon:
            root_node.label = self._max_label(labels)
          #  self.leaf.append(root_node)
            return
        """

        if len(info_gain_list_cat) > 0 and max(info_gain_list_cat) > cont_gain:
            flag = False
        else:
            flag = True
        
        if not flag:
            
            idx = info_gain_list_cat.index(max(info_gain_list_cat))
            ag = axises.pop(idx)
            _  = self.cat.pop(idx)
            root_node.ax = ag
            root_node.cont = None
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
        
        else:

            ag = cont_ax
            root_node.ax = ag
            ax_data = data[:, ag]
            ax_val = self.feature_list[ag][cont_idx]
            root_node.cont = ax_val

            left_idx = np.argwhere(ax_data <= ax_val).flatten()
            right_idx = np.argwhere(ax_data > ax_val).flatten()
            left_child_node = Node()
            right_child_node = Node()
            left_child_node.parent = root_node
            right_child_node.parent = root_node
            root_node.children[0] = left_child_node
            root_node.children[1] = right_child_node
            left_labels = labels[left_idx]
            right_labels = labels[right_idx]
            left_data = data[left_idx]
            right_data = data[right_idx]
            self._build(left_child_node, left_data, left_labels, axises)
            self._build(right_child_node, right_data, right_labels, axises)
        
    def train(self, data, labels):
        ndim = data.shape[1]
        axises = list(range(ndim))
        self._conti_feature(data)
        self._get_cont_feature_list(data)
        self._build(self.root, data, labels, axises)
    

    def predict(self, test, labels):
        counts = 0
        nevents = labels.size
        for i in range(nevents):
            temp_node = self.root
            while temp_node.children:
                if temp_node.cont is None:
                    ax = temp_node.ax
                    val = test[i, ax] 
                    temp_node = temp_node.children[val]
                else:
                    ax = temp_node.ax
                    val = test[i, ax]
                    if val <= temp_node.cont:
                        temp_node = temp_node.children[0]
                    else:
                        temp_node = temp_node.children[1]
            if temp_node.label == labels[i]:
                counts += 1
        accuracy = format(counts / nevents, '.5f')
        print("Model accuracy is:{}".format(accuracy))

if __name__ == "__main__":
    
    from sklearn.datasets import load_iris

    iris = load_iris()
    data, labels = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    model = DTree()
    print("Start training")
    model.train(X_train, y_train)

    print("Start predictin")
    model.predict(X_test, y_test)