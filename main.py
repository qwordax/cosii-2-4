import cv2 as cv
import numpy as np

from ideal import TWO, THREE, FOUR, FIVE, SEVEN
from network import Perceptron

EPSILON = 0.001

CHANCE = 0.1

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
    np.set_printoptions(precision=2, suppress=True)

    percept = Perceptron(36, 20, 5)

    training_data = [
        (binarize(TWO).T,   np.array([[1, 0, 0, 0, 0]]).T),
        (binarize(THREE).T, np.array([[0, 1, 0, 0, 0]]).T),
        (binarize(FOUR).T,  np.array([[0, 0, 1, 0, 0]]).T),
        (binarize(FIVE).T,  np.array([[0, 0, 0, 1, 0]]).T),
        (binarize(SEVEN).T, np.array([[0, 0, 0, 0, 1]]).T),
    ]

    n = len(training_data)

    # Total number of iterations.
    epochs = 0

    while True:
        np.random.shuffle(training_data)

        percept.gradient_descent(training_data)

        # Error function of the `percept`.
        delta = np.zeros((n, 1))

        for data in training_data:
            delta += (data[1]-percept.feed_forward(data[0])) ** 2

        if np.all(delta/n < EPSILON):
            break

        epochs += 1

    print(f'epochs: {epochs}')

    test_data = [
        noise(TWO, CHANCE),   noise(TWO, CHANCE),   noise(TWO, CHANCE),
        noise(THREE, CHANCE), noise(THREE, CHANCE), noise(THREE, CHANCE),
        noise(FOUR, CHANCE),  noise(FOUR, CHANCE),  noise(FOUR, CHANCE),
        noise(FIVE, CHANCE),  noise(FIVE, CHANCE),  noise(FIVE, CHANCE),
        noise(SEVEN, CHANCE), noise(SEVEN, CHANCE), noise(SEVEN, CHANCE),
    ]

    for i, test in enumerate(test_data):
        print(f'{i+1:2d}: {100*percept.feed_forward(binarize(test).T).T}')

    for i, image in enumerate(test_data):
        cv.imshow(f'Test {i+1}', cv.resize(
            image,
            (256, 256),
            interpolation=cv.INTER_NEAREST
        ))

    cv.imshow('2', cv.resize(
        TWO,
        (256, 256),
        interpolation=cv.INTER_NEAREST
    ))

    cv.imshow('3', cv.resize(
        THREE,
        (256, 256),
        interpolation=cv.INTER_NEAREST
    ))

    cv.imshow('4', cv.resize(
        FOUR,
        (256, 256),
        interpolation=cv.INTER_NEAREST
    ))

    cv.imshow('5', cv.resize(
        FIVE,
        (256, 256),
        interpolation=cv.INTER_NEAREST
    ))

    cv.imshow('7', cv.resize(
        SEVEN,
        (256, 256),
        interpolation=cv.INTER_NEAREST
    ))

    cv.waitKey(0)

if __name__ == '__main__':
    main()
