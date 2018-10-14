# this K-D Tree is speciallized for KNN search, so we have labels
# w/o delte, insert, update 
# But you can easily change it into normal K-D data structure
# reference for kNN search : https://zhuanlan.zhihu.com/p/23966698
# optimize code for python, as python does not support recursion well

import numpy as np
import numpy.linalg as LA

class Node():
	def __init__(self):
		self.data = None 
		self.label = None
		self.depth = 0
		self.left = None
		self.right = None
		self.parent = None

class KDTree():
	
	def __init__(self):
		self.root = Node()	
		self.nevents = 0
	
	def _split(self, train, labels, root_node):

		if train.size == 0:
			if root_node == root_node.parent.left:
				root_node.parent.left = None
			elif root_node == root_node.parent.right:
				root_node.parent.right = None
			root_node = None
			return

		ndim = train[0].size
		nsamples = train[:,0].size
		assert train[:,0].size == labels.size

		if train.shape[0] == 1:
			root_node.data = train.flatten()
			root_node.label = labels[0]
			return 

		j = root_node.depth
		i = j % ndim # split dimension
		m = nsamples // 2
		idx = np.argpartition(train[:,i], m)
		root_node.data = train[idx[m]]
		root_node.label = labels[idx[m]]
		Ldata = train[idx[:m]]
		Rdata = train[idx[m+1:]]
		Llabels = labels[idx[:m]]
		Rlabels = labels[idx[m+1:]]
		LNode = Node()
		RNode = Node()
		
		root_node.left = LNode
		LNode.data = Ldata
		LNode.label = Llabels
		LNode.depth = j + 1
		LNode.parent = root_node
		root_node.right = RNode
		RNode.data = Rdata
		RNode.label = Rlabels
		RNode.depth = j + 1	
		RNode.parent = root_node
		self._split(Ldata, Llabels, LNode)
		self._split(Rdata, Rlabels, RNode)
			

	def build(self, data, labels):
		self._split(data, labels, self.root)
		self.nevents = labels.size

	
	def _preorder(self, root_node):
		if root_node != None:
			print("data: {}".format(root_node.data))
		#	print("labels: {}".format(root_node.label))
			self._preorder(root_node.left)
			self._preorder(root_node.right)	
	
	def show(self):
		self._preorder(self.root)

	def _find_bottom(self, X, root_node):
		temp = root_node
		dim = X.size
		while temp != None:
			p = temp
			i = temp.depth % dim 
			if X[i] < temp.data[i]:
				temp = temp.left
			else:
				temp = temp.right
		return p

	# k_list is a list of Node, possible KNN of X
	def _max_dist(self, X, k_list, k=5):
		temp = 0
		idx = 0
		for i in range(len(k_list)):
			dist = LA.norm(X - k_list[i].data) 
			if dist > temp:
				temp = dist
				idx = i
		return idx, temp
				
	def _add_node(self, X, current_node, k_list, search_list, k=5):
		#current_node = bottom_node
		assert current_node not in search_list
		search_list.append(current_node)
		if len(k_list) < k:
			k_list.append(current_node)
		else:
			idx, max_d = self._max_dist(X, k_list, k)
			if LA.norm(X - current_node.data) < max_d:
				k_list[idx] = current_node	
		
				
	
	def _bottom_to_up(self, X, root_node, k_list, search_list, k=5):
		
		bottom_node = self._find_bottom(X, root_node)
		self._add_node(X, bottom_node, k_list, search_list, k)
		current_node = bottom_node
		for i in range(self.nevents):
			if current_node != self.root:
				temp_node = current_node
				current_node = current_node.parent
				if current_node in search_list:
					continue
				else:
					self._add_node(X, current_node, k_list, search_list, k)
					flag1 = 0
					flag2 = 0
					if (temp_node == current_node.left) and (current_node.right != None):
						flag1 = 1
					elif (temp_node == current_node.right) and (current_node.left != None):
						flag2 = 1
					else:
						continue

					i = current_node.depth % X.size 
					ax_dist = abs(current_node.data[i] - X[i])
					_, max_d = self._max_dist(X, k_list, k)
					
					if ((ax_dist < max_d) or (len(k_list) < k)) and (flag1 or flag2):
						if flag1: 
							return current_node.right
						else :
							return current_node.left
					else:
						continue
			else:
				return current_node	


	def kNN_points(self, X, k=5):
		search_list = []
		k_list = []

		current_node = self._bottom_to_up(X, self.root, k_list, search_list, k)

		for i in range(self.nevents):	
			if current_node != self.root:
				current_node = self._bottom_to_up(X, current_node, k_list, search_list, k)
			else:
				break

		return k_list
							

if __name__ == "__main__":
	
	test_data = np.array([[6.27, 5.50], [1.24, -2.86], [17.05, -12.79], [-6.88, -5.40],
				[-2.96, -0.50], [7.75, -22.68], [10.80, -5.03], [-4.60, -10.55], 
				[-4.96, 12.61], [1.75, 12.26], [15.31, -13.16], [7.83, 15.70], 
				[14.63, -0.35]])	

	test_labels = np.zeros(13)
	test_kdT = KDTree()
	test_kdT.build(test_data, test_labels)
	test_kdT.show()
	test = np.array([-1, -5])
	knn = test_kdT.kNN_points(test, k=3)
	for k in knn:
		print(k.data)
