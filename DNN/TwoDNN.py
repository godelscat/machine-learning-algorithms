import numpy as np 
import pandas as pd 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from TwoLayer import TwoLayer
import matplotlib.pyplot as plt

raw_data = pd.read_csv("../dataset/MNIST/train.csv")
data = raw_data.iloc[:,1:].values
labels = tf.keras.utils.to_categorical(raw_data["label"])

x_train, x_test, t_train, t_test = train_test_split(data, labels)
train_size = x_train.shape[0]
test_size = x_test.shape[0]

# super params
batch_size = 64
epochs = 20
nsamples = x_train.shape[0]
iter_num = epochs * nsamples / batch_size
learning_rate = 0.001
input_size = x_train.shape[1]

network = TwoLayer(input_size=input_size, hidden_size=64, output_size=10)
train_loss = []
train_acc = []
test_acc = []

for i in range(iter_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)
    
    for key in ['W1', 'b1', 'W2', 'b2']:
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss.append(loss)

    if i % int(train_size / batch_size) == 0:
        train_acc_val = network.accuracy(x_train, t_train)
        test_acc_val = network.accuracy(x_test, t_test)
        train_acc.append(train_acc_val)
        test_acc.append(test_acc_val)

ep = len(train_acc)
plt.plot(ep, train_acc, label='train acc')
plt.plot(ep, test_acc, label='test acc')
plt.xlabel('iter')
plt.ylabel('acc')
plt.savefig('train_test_acc.png')
