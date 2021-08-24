# -*- coding: utf-8 -*-
""" Module gathering all image filters which can be applied on a 2D numpy array

:author: Alain Giros
:organization: CNES
:created: 24 août 2021
"""
from functools import lru_cache

from scipy.signal import convolve2d
import numpy as np


# FIXME: find the right name of this filter
def filter_1(image_array: np.ndarray, window_size: int) -> np.ndarray:
    s1, s2 = np.shape(image_array)
    return image_array[int(s1 / 2 - window_size * s1 / 2):int(s1 / 2 + window_size * s1 / 2),
                       int(s2 / 2 - window_size * s2 / 2):int(s2 / 2 + window_size * s2 / 2)]


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


def desmooth(pixels, nx, ny):
    smoothed_pixels = funSmooth2(pixels, nx, ny)
    desmoothed_pixels = pixels - smoothed_pixels
    return desmoothed_pixels


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
