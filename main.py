import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import math
import zplane as zp

reverse_H_z = 0;

def rotate_90_left(matrix):
    n = len(matrix)
    m = len(matrix[0])
    newCoord = np.zeros(shape=(n, m))
    T = np.array([[0, 1], [-0, 0]])
    for x in range(n):
        for y in range(m):
            oldCoord = matrix[x][y]
            newCoord[x][y] = T.dot(oldCoord)

    return newCoord

def counter_warp(fe,fs):
    We = 2*np.pi * fe/fs
    Wa = 2*fs*np.tan(We/2)
    return Wa


def IIR_butterworth():
    pass

def IIR_chebychev_1():
    pass

def IIR_chebychev_2():
    pass

def IIR_elliptic():


if __name__ == '__main__':
    pass
