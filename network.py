import numpy as np

def sigmoid(x, deriv=False):
    if deriv is True:
        return sigmoid(x) / (1-sigmoid(x))

    return 1 / (1+np.exp(-x))

class Perceptron:
    def __init__(self, n, h, m, alpha=0.1, beta=0.1):
        self.v = 2*np.random.random((h, n)) - 1
        self.w = 2*np.random.random((m, h)) - 1

        self.q = 2*np.random.random((h, 1)) - 1
        self.t = 2*np.random.random((m, 1)) - 1

        self.alpha = alpha
        self.beta = beta

    def feed_forward(self, x):
        g = sigmoid(np.dot(self.v, x) + self.q)
        y = sigmoid(np.dot(self.w, g) + self.t)

        return y

    def gradient_descent(self, training_data):
        pass

    def back_propagation(self, x, y):
        current_g = sigmoid(np.dot(self.v, x) + self.q)
        current_y = sigmoid(np.dot(self.w, current_g) + self.t)

        delta_y = 2*(current_y-y) * sigmoid(current_y, deriv=True)

        nabla_v = np.dot(delta_y, current_g.T)
        nabla_q = delta_y

        delta_g = np.dot(self.w.T, delta_y) * sigmoid(current_g, deriv=True)

        nabla_w = np.dot(delta_g, x.T)
        nabla_t = delta_g

        return (nabla_v, nabla_w, nabla_q, nabla_t)
