# ref of KKT: http://liuhongjiang.github.io/tech/blog/2012/12/28/svm-smo/
# I use a slightly simply way to select the two alphas
# Final accuracy is around 95%, and takes about 2m48s in my computer
import numpy as np
import pandas as pd
import sys
sys.path.append("../perceptron/")

from perceptron import train_split

class SVM:

	def __init__(self):
		self.maxIter = 10000
		self.epsilon = 0.001 
		self.E = None # Eq(7.105) Ei
		self.kernel = None # linear kernel function
		self.b = 0 # bias b
		self.alpha = None # alpha in Eq(7.98)
		self.C = 10

	def _get_kernel(self, data):
		ndim = data.shape[0]	
		self.kernel = np.zeros((ndim, ndim))
		# as kernel function is symmetric
		for i in range(ndim):
			for j in range(i+1):
				self.kernel[i, j] =  np.dot(data[i], data[j])
				self.kernel[j, i] = self.kernel[i, j]

	# cal Eq(7.104)
	def _get_gi(self, labels, i):
		gi = np.sum(self.kernel[i] * labels * self.alpha)
		gi += self.b
		return gi

	# cal Eq(7.105)
	def _get_E(self, labels):
		bigMatrix = self.alpha * labels * self.kernel
		self.E = np.sum(bigMatrix, axis=1) + self.b - labels

	def _sign_func(self, train_data, train_labels, test_data):
		bigMax = np.sum(test_data * train_data, axis=1) 
		func = np.sum(self.alpha * train_labels * bigMax) + self.b	
		result = 1 if func >=0 else -1 
		return result

	def train(self, data, labels):
		ndim = labels.size
		self.alpha = np.zeros((ndim))
		self._get_kernel(data)
		self._get_E(labels)
		for k in range(self.maxIter):
			flag = True
			for i in range(ndim):
				alpha1 = self.alpha[i]
				E1 = self.E[i]
				y1 = labels[i]
				if (alpha1 < self.C and y1 * E1 < -self.epsilon) or (alpha1 > 0 and y1 * E1 > self.epsilon):
					if E1 > 0 :
						j  = np.argmin(self.E)
					else:
						j  = np.argmax(self.E)

					if j == i : 
						continue

					alpha2 = self.alpha[j]
					E2 = self.E[j]
					y2 = labels[j]

					eta = self.kernel[i,i] + self.kernel[j,j] - 2 * self.kernel[i,j]
					alpha2_unc = alpha2 + y2 * (E1 - E2) / eta
					if y1 == y2:
						L = max([0, alpha2 + alpha1 - self.C])
						H = min([self.C, alpha2 + alpha1])
					else:
						L = max([0, alpha2 - alpha1])
						H = min([self.C, self.C + alpha2 - alpha1])
					if alpha2_unc > H:
						alpha2_new = H
					elif alpha2_unc >= L:
						alpha2_new = alpha2_unc
					else:
						alpha2_new = L
					alpha1_new = alpha1 + y1 * y2 * (alpha2 - alpha2_new)

					# cal Eq(7.115) and Eq(7.116)
					b1_new = -E1 - y1 * self.kernel[i,i] * (alpha1_new - alpha1) - y2 * self.kernel[i,j] * (alpha2_new - alpha2) + self.b
					b2_new = -E2 - y1 * self.kernel[i,j] * (alpha1_new - alpha1) - y2 * self.kernel[j,j] * (alpha2_new - alpha2) + self.b

					# update alpha
					self.alpha[i] = alpha1_new
					self.alpha[j] = alpha2_new

					if alpha1_new > 0 and alpha1_new < self.C:
						self.b = b1_new
					elif alpha2_new > 0 and alpha2_new < self.C:
						self.b = b2_new
					else:
						self.b = (b1_new + b2_new) / 2
					
					# update Ei
					self.E[i] = self._get_gi(labels, i) - y1
					self.E[j] = self._get_gi(labels, j) - y2

					flag = False
					break
			# if KKT is satisfied, then stop
			if flag:
				break

	def predict(self, train_data, train_labels, test_data, test_labels):
		ndim = test_labels.size
		counts = 0
		for i in range(ndim):
			result = self._sign_func(train_data, train_labels, test_data[i])
			if result == test_labels[i]:
				counts += 1
		accuracy = format(counts / ndim, '.5f')	
		print("Model accuracy is : {}".format(accuracy))


if __name__ == "__main__":
	
	raw_data = pd.read_csv("../dataset/MNIST/binary_train.csv")
	labels = raw_data['label'].values
	data = raw_data.iloc[:,1:].values
	train_data, train_labels, test_data, test_labels = train_split(data, labels)

	model = SVM()
	print("start training")
	model.train(train_data, train_labels)
	print("start predicting")
	model.predict(train_data, train_labels, test_data, test_labels)
