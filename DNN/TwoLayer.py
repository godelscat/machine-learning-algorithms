import numpy as np 
from NaiveLayer import Affine, Relu, SoftmaxWithLoss
from collections import OrderedDict

class TwoLayer:
    def __init__(self, input_size, hidden_size, output_size):
        self.params = {}
        self.params['W1'] = 0.01 * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = 0.01 * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] =  Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        out = self.predict(x)
        out = self.lastLayer.forward(out, t)
        return out
         
    def accuracy(self, x, t):
        y = self.predict(x)
        assert t.ndim > 1
        temp1 = np.argmax(y, axis=1)
        temp2 = np.argmax(t, axis=1) 
        batch_size = t.shape[0]
        acc = np.sum(temp1 == temp2) / batch_size
        return acc

    def gradient(self, x, t):
        # first forward, update W and b
        _ = self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        for layer in layers[::-1]:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads