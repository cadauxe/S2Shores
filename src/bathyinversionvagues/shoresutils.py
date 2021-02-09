#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 3 10:12:00 2021

Module containing all functions common to waves and bathy estimation methods

@author: erwinbergsma
         gregoirethoumyre
         degoulromain

"""
import numpy as np
from skimage.transform import radon
from scipy.signal import convolve2d


# check array is not empty
def sc_all(array):
    for x in array.flat:
        if not x:
            return False
    return True


def funDetrend_2d(Z):
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

    return (Z_f)

def funSinoFFT(sino1, sino2, dx):
    '''
    Parameters
    ----------
    I : 2D np.array that contains NxMx2 structure.
        Typically these are two different images at the same location at a different moment in time.
    dx : Spatial resolution of the image I.
        DESCRIPTION. The default is 10 (sentinel)

    Returns
    -------
    out : Dictioinary with all infomation on the Radon spectrum
        DESCRIPTION.

    '''
    Nx = np.shape(sino1)[0]
    Fs = 1 / float(dx)
    kfft = np.arange(0, Fs / 2, Fs / Nx) + (Fs / (2 * Nx))

    sino1_fft = []
    sino2_fft = []

    for ii in range(0, sino1.shape[1]):
        Y1 = np.fft.fft(sino1[:, ii])
        Y2 = np.fft.fft(sino2[:, ii])

        if (len(Y1) > 1):
            Yout1 = Y1[0:int(np.ceil(Nx / 2))]
            Yout2 = Y2[0:int(np.ceil(Nx / 2))]
        else:
            Yout1 = Y1
            Yout2 = Y2

        sino1_fft.append(Yout1)
        sino2_fft.append(Yout2)

    sinoFFT = np.dstack((np.array(sino1_fft).transpose(), \
                         np.array(sino2_fft).transpose()))

    return sinoFFT, kfft, Nx

def funGetSpectralPeaks(Im, theta, params):
    sinogram1 = radon(funDetrend_2d(Im[:, :, 0]), theta=theta)
    sinogram2 = radon(funDetrend_2d(Im[:, :, 1]), theta=theta)

    sinoFFT, kfft, N = funSinoFFT(sinogram1, sinogram2, params['DX'])

    combAmpl = (np.abs(sinoFFT[:, :, 0]) ** 2 + np.abs(sinoFFT[:, :, 1]) ** 2) / (N ** 2)

    phase_shift = np.angle(sinoFFT[:, :, 1] * np.conj(sinoFFT[:, :, 0]))

    if params['UNWRAP_PHASE_SHIFT'] == False:
        # currently deactivated but we want this functionality:
        phase_shift = phase_shift
    else:
        phase_shift = (phase_shift + 2 * np.pi) % (2 * np.pi)

    # deep water limits:
    phi_deep = (2 * np.pi * params['DT']) / (np.sqrt(1 / (np.round(params['G']/(2*np.pi),2)* kfft))).squeeze()
    phi_deep = np.tile(phi_deep[:, np.newaxis], (1, theta.shape[0]))

    # shallow water limits:
    min_cel = np.sqrt(params['G'] * params['MIN_D'])
    phi_min = (2 * np.pi * params['DT'] * kfft * min_cel).squeeze()
    phi_min = np.tile(phi_min[:, np.newaxis], (1, theta.shape[0]))

    # Deep water limitation [if the wave travels faster that the deep-water limit we consider it non-physical]
    phase_shift[np.abs(phase_shift) > phi_deep] = 0

    # Minimal propagation speed; this depends on the Satellite; Venus or Sentinel 2
    phase_shift[np.abs(phase_shift) < phi_min] = 0

    return (combAmpl * phase_shift) / N, kfft, N, phase_shift


def DFT_fr(x, fr, fs):
    """
    Compute the discrete Fourier Transform of the 1D array x
    :param x: (array)
    """

    N = x.size
    n = np.arange(N)
    # k = n.reshape((N, 1))

    e = np.exp(-2j * np.pi * fr * n / fs)
    return np.dot(e, x)

def funConv2(x, y, mode='same'):
    '''


    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    mode : TYPE, optional
        DESCRIPTION. The default is 'same'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

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
    Nr = Nr + 1
    Nc = Nc + 1

    kr = 2 * Nr + 1
    kc = 2 * Nc + 1

    midr = Nr + 1
    midc = Nc + 1
    maxD = (Nr ** 2 + Nc ** 2) ** 0.5

    k = np.zeros((kr, kc))
    for irow in range(0, kr):
        for icol in range(0, kc):
            D = np.sqrt(((midr - irow) ** 2) + ((midc - icol) ** 2))
            k[irow, icol] = np.cos(D * np.pi / 2 / maxD);

    k = k / np.sum(k.ravel())
    # Perform convolution
    out = funConv2(mI, k, 'same')
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
    S = np.concatenate((np.tile(M[0, :], (nx, 1)).transpose(), \
                        M.transpose(), \
                        np.tile(M[-1, :], (nx, 1)).transpose()), axis=1).transpose()

    S = np.concatenate((np.tile(S[:, 0], (ny, 1)).transpose(), \
                        S, \
                        np.tile(S[:, -1], (ny, 1)).transpose()), axis=1)

    S = funSmoothc(S, nx - 1, ny - 1)

    return (S)

def funLinearC_k(nu, c,params):
    k = 2 * np.pi * nu  #angular wave number
    precision = params['D_PRECISION']
#    ct = 0
    w = c * k
    g = params['G']
    do = params['D_INIT']
    d = c ** 2 / g

    while (abs(do - d) > precision):
        #ct = ct + 1
        do = d
        dispe = w ** 2 - (g * k * np.tanh(k * d))
        fdispe = -g * (k ** 2) / (np.cosh(k * d) ** 2)
        d = d - (dispe / fdispe)

    return (d)