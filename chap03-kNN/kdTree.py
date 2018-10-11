# this K-D Tree is speciallized for KNN search, so we have labels
# But you can change it into normal K-D data structure
# reference for kNN search : https://zhuanlan.zhihu.com/p/23966698

import numpy as np

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
			root_node.data = train
			root_node.label = labels
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

	
	def _inorder(self, root_node):
		if root_node != None:
			print("data: {}".format(root_node.data))
		#	print("labels: {}".format(root_node.label))
			self._inorder(root_node.left)
			self._inorder(root_node.right)	
	
	def show(self):
		self._inorder(self.root)

if __name__ == "__main__":
	
	test_data = np.array([[2,3], [5,4], [9,6], [4,7], [8,1], [7,2]])	
	test_labels = np.zeros(6)	
	test_kdT = KDTree()
	test_kdT.build(test_data, test_labels)
	test_kdT.show()
