import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import math
import zplane as zp

fs = 1600

ABERRATIONS     = "C:\\Users\\15142\\Desktop\\S5_APP4\\Digital_filter_class_2__IIR_filters\\images\\goldhill_aberrations.npy"
GOLDHILL_BRUIT  = "C:\\Users\\15142\\Desktop\\S5_APP4\\Digital_filter_class_2__IIR_filters\\images\\goldhill_bruit.npy"
GOLDHILL_ROTATE = "C:\\Users\\15142\\Desktop\\S5_APP4\\Digital_filter_class_2__IIR_filters\\images\\goldhill_rotate.png"
IMG_COMPLETE    = "C:\\Users\\15142\\Desktop\\S5_APP4\\Digital_filter_class_2__IIR_filters\\images\\image_complete.npy"

def get_png_array(filename):
    matplotlib.pyplot.gray()
    img_couleur = matplotlib.image.imread(filename)
    img_gris = np.mean(img_couleur, -1)
    return img_gris

def get_np_array(filename):
    img_array = np.load(filename)
    return img_array

reverse_H_z = 0

def rotate_90_left(matrix):
    n = len(matrix)
    m = len(matrix[0])
    newImg = np.zeros(shape=(n, m))
    T = np.array([[0, 1], [-1, 0]])
    for x in range(n):
        for y in range(m):
            value = matrix[x][y]
            oldCoord = np.array([x, y])
            newCoord = T.dot(oldCoord)
            newImg[newCoord[0], newCoord[1]] = value

    return newImg

def counter_warp(fe,fs):
    We = 2*np.pi * fe/fs
    Wa = 2*fs*np.tan(We/2)
    print(Wa/(2*np.pi))
    return Wa/(2*np.pi)

def get_ws_wp(band_pass, band_cut):
    ws = 2*np.pi*band_pass/fs
    wp = 2*np.pi*band_cut/fs
    return ws, wp

def IIR_butterworth():
    band_pass = 500
    band_cut = 750
    gain_pass = 0.2
    gain_cut = 60
    ws, wp = get_ws_wp(band_pass, band_cut)

    order, wn = signal.buttord(wp, ws, gain_pass, gain_cut)

    b, a = signal.butter(order, wn)
    W, H = signal.freqz(b=b, a=a)

    return W, H

def IIR_chebychev_1():
    band_pass = 500
    band_cut = 750
    gain_pass = 0.2
    gain_cut = 60
    ws, wp = get_ws_wp(band_pass, band_cut)
    order, wn = signal.cheb1ord(wp, ws, gain_pass, gain_cut)
    rp = 0.0000001

    b, a = signal.cheby1(order, gain_pass, wn*fs/(2*np.pi), btype='low', output='ba', fs=fs)

    W, H = signal.freqz(b=b, a=a)

    return W, H

def IIR_chebychev_2():
    band_pass = 500
    band_cut = 750
    gain_pass = 0.2
    gain_cut = 60
    ws, wp = get_ws_wp(band_pass, band_cut)
    order, wn = signal.cheb2ord(wp, ws, gain_pass, gain_cut)

    rp = 0.0001

    b, a = signal.cheby2(order, gain_pass, wn*fs/(2*np.pi), btype='lowpass', analog=False, output='ba', fs=fs)

    W, H = signal.freqz(b=b, a=a)

    return W, H

def IIR_elliptic():
    band_pass = 500
    band_cut = 750
    gain_pass = 0.2
    gain_cut = 60
    ws, wp = get_ws_wp(band_pass, band_cut)
    print(wp,ws)
    order, wn = signal.ellipord(wp, ws, gain_pass, gain_cut, fs)
    b, a = signal.ellip(order, gain_pass, gain_cut, (wn*fs/(2*np.pi)), btype='lowpass', output='ba', fs=fs)

    W, H = signal.freqz(b=b, a=a)

    return W, H

def home_made_IIR(image):
    a = 2*fs
    b = np.sqrt(2)
    wc = 2*np.pi*500/1600
    c = 2*1600*np.tan(wc/2)
    num = np.array([(wc/(a**2 + a*b + c)), 2*(wc/(a**2 + a*b + c)), (wc/(a**2 + a*b + c))])
    denum = np.array([1, ( (2*c-2*(a**2))/(a**2 + a*b + c) ), ( (a**2 + c - a*b)/(a**2 + a*b + c) )])

    response = signal.lfilter(num, denum, image)

    return response


if __name__ == '__main__':
    # W, H = IIR_elliptic()
    # plt.plot((W*fs/(2*np.pi)), np.abs(H))
    # plt.show()

    img = get_np_array(GOLDHILL_BRUIT)



    plt.imshow(img, interpolation='nearest')
    plt.show()

    # img = get_png_array(GOLDHILL_ROTATE)
    # img = rotate_90_left(img)
    #
    # plt.imshow(img, interpolation='nearest')
    # plt.show()

