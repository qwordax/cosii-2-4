import numpy as np

TWO = np.array([
    [255,   0,   0,   0,   0, 255],
    [  0,   0, 255, 255,   0,   0],
    [255, 255, 255, 255,   0,   0],
    [255, 255, 255,   0,   0, 255],
    [255,   0,   0,   0, 255, 255],
    [  0,   0,   0,   0,   0,   0],
], dtype=np.uint8)
"""
An ideal pattern representing the digit `2`.
"""

THREE = np.array([
    [255,   0,   0,   0,   0, 255],
    [  0,   0, 255, 255,   0,   0],
    [255, 255, 255,   0,   0,   0],
    [255, 255, 255, 255, 255,   0],
    [  0,   0, 255, 255,   0,   0],
    [255,   0,   0,   0,   0, 255],
], dtype=np.uint8)
"""
An ideal pattern representing the digit `3`.
"""

FOUR = np.array([
    [255,   0, 255, 255,   0,   0],
    [255,   0, 255, 255,   0,   0],
    [  0,   0, 255, 255,   0, 255],
    [  0,   0,   0,   0,   0, 255],
    [255, 255, 255,   0,   0, 255],
    [255, 255, 255,   0, 255, 255],
], dtype=np.uint8)
"""
An ideal pattern representing the digit `4`.
"""

FIVE = np.array([
    [255,   0,   0,   0,   0,   0],
    [  0,   0, 255, 255, 255, 255],
    [  0,   0,   0,   0,   0, 255],
    [255, 255, 255, 255,   0,   0],
    [  0,   0, 255, 255,   0,   0],
    [255,   0,   0,   0,   0, 255],
], dtype=np.uint8)
"""
An ideal pattern representing the digit `5`.
"""

SEVEN = np.array([
    [255,   0,   0,   0,   0,   0],
    [  0,   0, 255, 255,   0,   0],
    [255, 255, 255, 255,   0, 255],
    [255, 255, 255,   0,   0, 255],
    [255, 255, 255,   0,   0, 255],
    [255, 255, 255,   0, 255, 255],
], dtype=np.uint8)
"""
An ideal pattern representing the digit `7`.
"""
