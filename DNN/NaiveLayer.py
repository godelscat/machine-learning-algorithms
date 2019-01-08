import numpy as np 

# mini-batch ont-hot label cross entropy error
# y, t shape (N, )
def cross_entropy_error(y, t):
    # at least one sample, one possible label
    assert y.ndim > 1  

    batch_size = y.shape[0]
    small_num = 1e-7
    out = -np.sum(t * np.log(y + small_num)) / batch_size
    return out 

#mini-batch one-hot label softmax function
def softmax(x):
    batch_size = x.shape[0]
    x_max = np.max(x, axis=1)
    x_max = x_max.reshape(batch_size, -1)
    x_exp = np.exp(x - x_max) # avoid overflow
    sum_exp_x = np.sum(x_exp, axis=1)
    sum_exp_x = sum_exp_x.reshape(batch_size, -1)
    y = x_exp / sum_exp_x
    return y

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = None
        self.y = None
        out = x * y
        return out 

    def backward(self, dout):
        dx = self.y * dout 
        dy = self.x * dout 
        return dx, dy 

class AddLayer:
    def __init__(self):
        pass 
    
    def forward(self, x, y):
        self.x = x 
        self.y = y
        out = x + y
        return out 
    
    def backward(self, dout):
        dx = dout 
        dy = dout 
        return dx, dy 

class Relu:
    def __init__(self):
        self.mask = None 

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out 
    
    def backward(self, dout):
        dout[self.mask] = 0 
        dx = dout 
        return dx 

class Sigmoid:
    def __init__(self):
        self.out = None 
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out 
        return out 
    
    def backward(self, dout):
        dx = dout * (1 - self.out) * self.out 
        return dx

class Affine:
    def __init__(self, w, b):
        self.W = w
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out 
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, t)
        return self.loss

    def backward(self, dout):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx