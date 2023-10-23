import numpy as np

def sigmoid(x, deriv=False):
    if deriv is True:
        return sigmoid(x) / (1-sigmoid(x))

    return 1 / (1+np.exp(-x))

class Perceptron:
    def __init__(self, x, g, y, alpha, beta):
        self.v = 2*np.random.random((g, x)) - 1
        self.w = 2*np.random.random((y, g)) - 1

        self.q = 2*np.random.random((g, 1)) - 1
        self.t = 2*np.random.random((y, 1)) - 1

        self.alpha = alpha
        self.beta = beta

    def feed_forward(self, x):
        g = sigmoid(np.dot(self.v, x) + self.q)
        y = sigmoid(np.dot(self.w, g) + self.t)

        return y
