# Following WenDesi/lihang_book_algorithm, we use MNIST dataset
# However, here we label odd number -1, even number +1
import numpy as np
import pandas as pd

def train_split(data, labels, train_ratio=0.3):
	assert data[:,0].size == labels.size
	nevents = labels.size
	idx = np.arange(nevents)
	np.random.shuffle(idx)
	data = data[idx, :]	
	labels = labels[idx]
	train_split = int(nevents*(1-train_ratio)+1)
	train_data = data[:train_split, :]
	test_data = data[train_split:, :]
	train_labels = labels[:train_split]
	test_labels = labels[train_split:]
	return train_data, train_labels, test_data, test_labels
	
class Perceptron():

	def __init__(self, eta=0.00001):
		self.learning_rate = eta 

	def train(self, data, labels):
		assert data[:,0].size == labels.size
		nevents = labels.size
		runtime = 10000
		run = 0
		# initialize w and b to 0
		weight = np.zeros(data[0,:].size)
		bias = 0
		while True:
			idx = np.random.randint(0, nevents)
			temp = labels[idx] * (np.dot(weight, data[idx,:]) + bias)
			if temp <= 0 :
				weight = weight + self.learning_rate * labels[idx] * data[idx, :]
				bias = bias + self.learning_rate * labels[idx]	
				run += 1
			if run >= runtime:
				print("up to max runtime")
				break
		return weight, bias

	def predict(self, test, labels, weight, bias):
		assert test[:,0].size == labels.size
		events = labels.size
		correct_label = 0
		for i in range(events):
			flag = labels[i] * (np.dot(weight, test[i,:]) + bias)
			if flag > 0:
				correct_label += 1
		accuracy = format(correct_label / events, '.5f')
		print("Model accuracy is {}.".format(accuracy))

if __name__ == "__main__":

	raw_data = pd.read_csv("../dataset/MNIST/binary_train.csv")
	labels = raw_data['label'].values
	data = raw_data.iloc[:,1:].values
	train_data, train_labels, test_data, test_labels = train_split(data,
labels)
	print(train_data.shape, test_data.shape)
	model = Perceptron()
	weight, bias = model.train(train_data, train_labels)
	model.predict(test_data, test_labels, weight, bias)
