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
    matplotlib.pyplot.gray()
    img_array = np.load(filename)
    return img_array


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


def H_z_reverse(image):
    num = np.array([-0.99, -0.99, 0.8])
    denum = np.array([0.9 * np.exp(1j * np.pi / 2), 0.9 * np.exp(-1j * np.pi / 2), 0.95 * np.exp(1j * np.pi / 8),
                      0.95 * np.exp(-1j * np.pi / 8)])
    b = np.poly(num)
    a = np.poly(denum)

    zp.zplane(b, a)
    W, H = signal.freqz(b=b, a=a)

    response = signal.lfilter(b, a, image)

    return response

def retirerBruitHF(image):
    f_cd = 500 # Hz
    f_s = 1600 # Hz
    w_d = 2 * np.pi * (f_cd/f_s)
    w_a = 2 * f_s * np.tan(w_d/2)
    a = 2*f_s/w_a
    b = np.sqrt(2)
    c = 1
    num = [(c/(a*a+a*b+c)), (2*c/(a*a+a*b+c)), (c/(a*a+a*b+c))]
    den = [1, ((2*c - 2*a*a)/(a*a+a*b+c)), ((a*a-a*b+c)/(a*a+a*b+c))]
    image = signal.lfilter(num,den,image)
    return image

# def counter_warp(fe,fs):
#     We = 2*np.pi * fe/fs
#     Wa = 2*fs*np.tan(We/2)
#     print(Wa/(2*np.pi))
#     return Wa/(2*np.pi)

def get_ws_wp(band_pass, band_cut):
    ws = 2*np.pi*band_pass/fs
    wp = 2*np.pi*band_cut/fs
    return ws, wp

def IIR_butterworth(image):
    band_pass = 500
    band_cut = 750
    gain_pass = 0.2
    gain_cut = 60
    ws, wp = get_ws_wp(band_pass, band_cut)

    order, wn = signal.buttord(wp, ws, gain_pass, gain_cut)

    b, a = signal.butter(order, wn)
    W, H = signal.freqz(b=b, a=a)
    zp.zplane(b, a)
    response = signal.lfilter(b, a, image)

    return response

def IIR_chebychev_1(image):
    band_pass = 500
    band_cut = 750
    gain_pass = 0.2
    gain_cut = 60
    ws, wp = get_ws_wp(band_pass, band_cut)
    order, wn = signal.cheb1ord(wp, ws, gain_pass, gain_cut)
    rp = 0.0000001

    b, a = signal.cheby1(order, gain_pass, wn*fs/(2*np.pi), btype='low', output='ba', fs=fs)
    zp.zplane(b, a)
    W, H = signal.freqz(b=b, a=a)

    response = signal.lfilter(b, a, image)

    return response


def IIR_chebychev_2(image):
    band_pass = 500
    band_cut = 750
    gain_pass = 0.2
    gain_cut = 60
    ws, wp = get_ws_wp(band_pass, band_cut)
    order, wn = signal.cheb2ord(wp, ws, gain_pass, gain_cut)

    rp = 0.0001

    b, a = signal.cheby2(order, gain_pass, wn*fs/(2*np.pi), btype='lowpass', analog=False, output='ba', fs=fs)
    zp.zplane(b, a)
    W, H = signal.freqz(b=b, a=a)

    response = signal.lfilter(b, a, image)

    return response


def IIR_elliptic(image):
    band_pass = 500
    band_cut = 750
    gain_pass = 0.2
    gain_cut = 60
    ws, wp = get_ws_wp(band_pass, band_cut)
    print(wp,ws)
    order, wn = signal.ellipord(wp, ws, gain_pass, gain_cut, fs)
    b, a = signal.ellip(order, gain_pass, gain_cut, (wn*fs/(2*np.pi)), btype='lowpass', output='ba', fs=fs)
    zp.zplane(b, a)
    W, H = signal.freqz(b=b, a=a)

    response = signal.lfilter(b, a, image)

    return response

def home_made_IIR(image):
    wd = 2 * np.pi * 500 / 1600     # frequence normalisée
    wa = 2*1600*np.tan(wd/2)        # fréquence normalisée ajustée
    a = 2*fs/wa
    b = np.sqrt(2)
    c = 1

    num = np.array([(c/(a**2 + a*b + c)), 2*(c/(a**2 + a*b + c)), (c/(a**2 + a*b + c))])
    denum = np.array([1, ( (2*c-2*(a**2))/(a**2 + a*b + c) ), ( (a**2 + c - a*b)/(a**2 + a*b + c) )])

    zp.zplane(num, denum)

    response = signal.lfilter(num, denum, image)

    return response

def compress(image):


    cov = np.cov(image)
    eigenvalues, eigenvector = np.linalg.eig(cov)

    # print("eigenV =\n", eigenvector, "LEN:\n", len(eigenvector))
    trans_eigenvec = np.transpose(eigenvector)
    newimage = trans_eigenvec.dot(image)

    vec_len = len(newimage[0])
    img_len = len(newimage)
    for i in range(img_len):
        if i<img_len*30//100:
            continue
        else:
            newimage[i] = np.zeros(vec_len)
    return newimage, eigenvector

def reverse_compress(img, eigenvector):
    O_G = eigenvector.dot(img)
    return O_G

def test_retirer_bruit():
    img = get_np_array(GOLDHILL_BRUIT)
    img = home_made_IIR(img)
    plt.imshow(img, interpolation='nearest')
    plt.show()

def test_rotate():
    img = get_png_array(GOLDHILL_ROTATE)
    img = rotate_90_left(img)
    plt.imshow(img, interpolation='nearest')
    plt.show()

def test_retirer_abheration():
    img = get_np_array(ABERRATIONS)
    img = H_z_reverse(img)
    plt.imshow(img, interpolation='nearest')
    plt.show()

def test_compress():
    img = get_png_array(GOLDHILL_ROTATE)
    img, eigen = compress(img)
    plt.imshow(img, interpolation='nearest')
    plt.show()
    img = reverse_compress(img, eigen)
    plt.imshow(img, interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    # test_retirer_abheration()
    test_compress()
    # W, H = IIR_elliptic()
    # plt.plot((W*fs/(2*np.pi)), np.abs(H))
    # plt.show()

    # img = get_np_array(GOLDHILL_BRUIT)
    #
    # plt.imshow(img, interpolation='nearest')
    # plt.show()
    #
    # img = IIR_elliptic(img)



    #
    # img = reverse_compress(img, eigen)
    # plt.imshow(img, interpolation='nearest')
    # plt.show()
    #
    # img = IIR_elliptic(img)
    # plt.imshow(img, interpolation='nearest')
    # plt.show()






