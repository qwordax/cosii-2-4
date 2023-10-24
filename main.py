import numpy as np

from ideal import TWO, THREE, FOUR, FIVE, SEVEN
from network import Perceptron

def noize(image, chance):
    result = image.copy()

    for i, x in np.ndenumerate(result):
        if np.random.random() < chance:
            result[i] = 255-x

    return result

def binarize(image):
    result = image.copy()

    for i, x in np.ndenumerate(result):
        result[i] = (255-x) / 255

    return result.reshape((result.size, 1))

def main():
    percept = Perceptron(36, 20, 5)

    training_data = [
        (binarize(TWO),   np.array([[1, 0, 0, 0, 0]]).T),
        (binarize(THREE), np.array([[0, 1, 0, 0, 0]]).T),
        (binarize(FOUR),  np.array([[0, 0, 1, 0, 0]]).T),
        (binarize(FIVE),  np.array([[0, 0, 0, 1, 0]]).T),
        (binarize(SEVEN), np.array([[0, 0, 0, 0, 1]]).T),
    ]

if __name__ == '__main__':
    main()
