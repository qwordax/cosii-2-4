import numpy as np

from ideal import TWO, THREE, FOUR, FIVE, SEVEN
from network import Perceptron

def noize(image, delta):
    result = image.copy()

    for i, x in np.ndenumerate(result):
        if x == 255:
            x -= np.random.randint(0, delta)
        else:
            x += np.random.randint(0, delta)

        result[i] = x

    return result

def binarize(image):
    result = image.copy()

    for i, x in np.ndenumerate(result):
        result[i] = (255-x) / 255

    return result.reshape((result.size, 1))

def main():
    percept = Perceptron(36, 20, 5)

if __name__ == '__main__':
    main()
