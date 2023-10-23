import numpy as np

class Perceptron:
    def __init__(self, x, g, y):
        self.v = 2*np.random.random((g, x)) - 1
        self.w = 2*np.random.random((y, g)) - 1

        self.q = 2*np.random.random((1, g)) - 1
        self.t = 2*np.random.random((1, y)) - 1
