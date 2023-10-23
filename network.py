import numpy as np

def sigmoid(x, deriv=False):
    if deriv is True:
        return sigmoid(x) / (1-sigmoid(x))

    return 1 / (1+np.exp(-x))

class Perceptron:
    def __init__(self, x, g, y, alpha, beta):
        self.v = 2*np.random.random((g, x)) - 1
        self.w = 2*np.random.random((y, g)) - 1

        self.q = 2*np.random.random((1, g)) - 1
        self.t = 2*np.random.random((1, y)) - 1

        self.alpha = alpha
        self.beta = beta
