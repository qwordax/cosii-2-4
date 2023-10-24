import numpy as np

from ideal import TWO, THREE, FOUR, FIVE, SEVEN
from network import Perceptron

def noise(image, chance):
    """
    Returns a noisy `image` determined by `chance`.
    """
    result = image.copy()

    for i, x in np.ndenumerate(result):
        if np.random.random() < chance:
            result[i] = 255-x

    return result

def binarize(image):
    """
    Implements an `image` binarization.
    """
    result = image.copy()

    for i, x in np.ndenumerate(result):
        result[i] = (255-x) / 255

    return result.reshape((1, result.size))

def main():
    """
    The main function of the program.
    """
    percept = Perceptron(36, 20, 5)

    training_data = [
        (binarize(TWO).T,   np.array([[1, 0, 0, 0, 0]]).T),
        (binarize(THREE).T, np.array([[0, 1, 0, 0, 0]]).T),
        (binarize(FOUR).T,  np.array([[0, 0, 1, 0, 0]]).T),
        (binarize(FIVE).T,  np.array([[0, 0, 0, 1, 0]]).T),
        (binarize(SEVEN).T, np.array([[0, 0, 0, 0, 1]]).T),
    ]

if __name__ == '__main__':
    main()
