# -*- coding: utf-8 -*-
"""
Created on Wed Feb 3 10:12:00 2021

Module containing all functions common to waves and bathy estimation methods

@author: erwinbergsma
         gregoirethoumyre
         degoulromain

"""
from functools import lru_cache
from scipy.signal import convolve2d
from scipy.signal import detrend
from scipy.signal import fftconvolve

import numpy as np


def funDetrend_2d(Z):
    """
    Performs detrending on a matrix
    :param Z:
    :return:
    """
    M = Z.shape[1]
    N = Z.shape[0]

    XX, YY = np.meshgrid(range(0, M), range(0, N))
    Xcolv = XX.flatten()
    Ycolv = YY.flatten()
    Zcolv = Z.flatten()

    Mp = np.zeros((len(Ycolv), 3))
    Const = np.ones(len(Xcolv))
    Mp[:, 0] = Xcolv
    Mp[:, 1] = Ycolv
    Mp[:, 2] = Const

    inv_Mp = np.linalg.pinv(Mp)

    Coef = np.dot(inv_Mp, Zcolv)
    Coef = np.asarray(Coef)

    XCoeff = Coef[0]
    YCoeff = Coef[1]
    CCoeff = Coef[2]

    Z_p = XCoeff * XX + YCoeff * YY + CCoeff
    Z_f = Z - Z_p

    return Z_f


def get_unity_roots(number_of_roots: int, fr: np.ndarray) -> np.ndarray:
    """
    Compute complex roots of the unity for some frequencies
    :param number_of_roots: Number of unity roots to compute, starting from 0
    :param fr: 1D array of normalized frequencies where roots are needed
    :returns: number_of_roots complex roots of the unity corresponding to fr frequencies
    """
    n = np.arange(number_of_roots)
    return np.exp(-2j * np.pi * fr * n)


def DFT_fr(x: np.ndarray, unity_roots: np.ndarray):
    """ Compute the discrete Fourier Transform of a 1D array

    :param np.ndarray x: 1D array containing the signal
    :param np.ndarray
    """
    # FIXME: used to interpolate spectrum, but seems incorrect. Use zero padding instead ?
    return np.dot(unity_roots, x)


@lru_cache()
def get_smoothing_kernel(Nr: int, Nc: int) -> np.ndarray:
    '''

    Parameters
    ----------
    Nr : TYPE
        DESCRIPTION.
    Nc : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    '''
    % SMOOTHC.M: Smooths matrix data, cosine taper.
    % MO=SMOOTHC(MI,Nr,Nc) smooths the data in MI
    % using a cosine taper over 2*N+1 successive points, Nr, Nc points on
    % each side of the current point.
    %
    % Nr - number of points used to smooth rows
    % Nc - number of points to smooth columns
    % Outputs: kernel to be used for smoothing
    %
    %
    '''

    # Determine convolution kernel k
    kr = 2 * Nr + 1
    kc = 2 * Nc + 1
    midr = Nr + 1
    midc = Nc + 1
    maxD = (Nr ** 2 + Nc ** 2) ** 0.5

    k = np.zeros((kr, kc))
    for irow in range(0, kr):
        for icol in range(0, kc):
            D = np.sqrt(((midr - irow) ** 2) + ((midc - icol) ** 2))
            k[irow, icol] = np.cos(D * np.pi / 2 / maxD)

    return k / np.sum(k.ravel())


def funSmoothc(mI, Nr, Nc):
    '''

    Parameters
    ----------
    mI : TYPE
        DESCRIPTION.
    Nr : TYPE
        DESCRIPTION.
    Nc : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    '''
    % SMOOTHC.M: Smooths matrix data, cosine taper.
    % MO=SMOOTHC(MI,Nr,Nc) smooths the data in MI
    % using a cosine taper over 2*N+1 successive points, Nr, Nc points on
    % each side of the current point.
    %
    % Inputs: mI - original matrix
    % Nr - number of points used to smooth rows
    % Nc - number of points to smooth columns
    % Outputs:mO - smoothed version of original matrix
    %
    %
    '''
    # Determine convolution kernel k
    k = get_smoothing_kernel(Nr, Nc)
    # Perform convolution
    out = np.rot90(convolve2d(np.rot90(mI, 2), np.rot90(k, 2), mode='same'), 2)
    return out[Nr:-Nr, Nc:-Nc]


def funSmooth2(M, nx, ny):
    '''
    Parameters
    ----------
    M : TYPE
        DESCRIPTION.
    nx : TYPE
        DESCRIPTION.
    ny : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    S = np.concatenate((np.tile(M[0, :], (nx, 1)).transpose(),
                        M.transpose(),
                        np.tile(M[-1, :], (nx, 1)).transpose()), axis=1).transpose()

    T = np.concatenate((np.tile(S[:, 0], (ny, 1)).transpose(),
                        S,
                        np.tile(S[:, -1], (ny, 1)).transpose()), axis=1)

    return funSmoothc(T, nx, ny)


def fft_filtering(simg, spatial_resolution, T_max, T_min, gravity):
    """
    Compute the fft filtering of a subtile
    :param simg:(np.array) the given sequence of images to filter
    :param spatial_resolution: (int) sampling resolution (default 10 meters on Sentinel 2)
    :param T_max:(int) Max wave periode
    :param T_min:(int) Min wave periode
    :return: simg_filtered:
    """
    flag = 0
    n, m, c = simg.shape
    kx = np.fft.fftshift(np.fft.fftfreq(n, spatial_resolution))
    ky = np.fft.fftshift(np.fft.fftfreq(m, spatial_resolution))
    kx = np.repeat(np.reshape(kx, (n, 1)), m, axis=1)
    ky = np.repeat(np.reshape(ky, (1, m)), n, axis=0)
    # TODO: rely on wavenumber_offshore() function (later on remove Tmax and Tmin arguments)
    threshold_min = 2. * np.pi / (gravity * T_max ** 2)
    threshold_max = 2. * np.pi / (gravity * T_min ** 2)
    simg_filtered = np.zeros(simg.shape)
    kr = np.sqrt(kx ** 2 + ky ** 2)
    kr[kr < threshold_min] = 0
    kr[kr > threshold_max] = 0
    boolKr = (kr > 0)
    for channel in range(c):
        r = simg[:, :, channel]
        r = detrend(detrend(r, axis=1), axis=0)
        fftr = np.fft.fft2(r)
        energy_r = np.fft.fftshift(fftr)
        energy_r *= boolKr
        max_energy = np.max(np.abs(energy_r))
        if max_energy > 3 or max_energy < 0.01:
            flag = 1
            simg_filtered[:, :, channel] = np.real(np.fft.ifft2(np.fft.ifftshift(energy_r)))
    return simg_filtered, flag


def cross_correlation(A, B):
    """
    Compute the correlation of each line of A with each line of B
    This function is faster than using np.corrcoef which computes correlation beetween A&A,A&B,B&A,B&B
    :param A (np.array) : matrix A
    :param B (np.array) : matrix B
    :return: sub_tile (np.array) : cross correlation matrix
    """
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def filter_mean(signal: np.ndarray, size_window: int) -> np.ndarray:
    """ Run average filter on a signal

    :param signal: np.array in one dimension
    :param size_window: size of the averaging window
    :return: the mean filtered signal
    :reaises ValueError: when the signal is smaller than twice the window size
    """
    if len(signal) < 2 * size_window:
        raise ValueError("time serie is too small compared to the window")

    padded_signal = np.concatenate((np.full(size_window, np.mean(signal[:size_window])),
                                    signal,
                                    np.full(size_window,
                                            np.mean(signal[-(size_window + 1):]))),
                                   axis=0)
    return np.convolve(padded_signal, np.ones(2 * size_window + 1) / (2 * size_window + 1),
                       'valid')


def normxcorr2(template, image, mode="full"):
    ########################################################################################
    # Author: Ujash Joshi, University of Toronto, 2017                                     #
    # Based on Octave implementation by: Benjamin Eltzner, 2014 <b.eltzner@gmx.de>         #
    # Octave/Matlab normxcorr2 implementation in python 3.5                                #
    # Details:                                                                             #
    # Normalized cross-correlation. Similiar results upto 3 significant digits.            #
    # https://github.com/Sabrewarrior/normxcorr2-python/master/norxcorr2.py                #
    # http://lordsabre.blogspot.ca/2017/09/matlab-normxcorr2-implemented-in-python.html    #
    ########################################################################################
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the �full� output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """
    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
        len([i for i in range(np.ndim(template)) if
             template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
        np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return out
