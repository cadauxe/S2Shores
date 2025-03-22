# -*- coding: utf-8 -*-
""" Module gathering all image filters which can be applied on a 2D numpy array

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 24 août 2021

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
from functools import lru_cache

import numpy as np
from scipy.signal import convolve2d

from .numpy_utils import circular_mask, gaussian_mask


def clipping(image_array: np.ndarray, ratio_size: float) -> np.ndarray:
    """ Performs clipping of the edges

    :param image_array: entry image
    :param ratio_size: ratio of the image to keep (1 is the full image)
    :returns: clipped image
    """
    s1, s2 = np.shape(image_array)
    return image_array[int(s1 / 2 - ratio_size * s1 / 2):int(s1 / 2 + ratio_size * s1 / 2),
                       int(s2 / 2 - ratio_size * s2 / 2):int(s2 / 2 + ratio_size * s2 / 2)]


def detrend(image_array: np.ndarray) -> np.ndarray:
    """ Performs detrending on a matrix

    :param image_array: entry image
    :returns: detrended image
    """
    shape1 = image_array.shape[1]
    shape2 = image_array.shape[0]

    xx, yy = np.meshgrid(range(0, shape1), range(0, shape2))
    xcolv = xx.flatten()
    ycolv = yy.flatten()
    zcolv = image_array.flatten()

    mp = np.zeros((len(ycolv), 3))
    const = np.ones(len(xcolv))
    mp[:, 0] = xcolv
    mp[:, 1] = ycolv
    mp[:, 2] = const

    inv_mp = np.linalg.pinv(mp)
    coef = np.dot(inv_mp, zcolv)
    coef = np.asarray(coef)
    z_p = coef[0] * xx + coef[1] * yy + coef[2]
    z_f = image_array - z_p
    return z_f


@lru_cache()
def get_smoothing_kernel(Nr: int, Nc: int) -> np.ndarray:
    """

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
    """

    # Determine convolution kernel k
    kernel_rows = 2 * Nr + 1
    kernel_columns = 2 * Nc + 1
    midr = Nr + 1
    midc = Nc + 1
    maxD = (Nr ** 2 + Nc ** 2) ** 0.5

    k = np.zeros((kernel_rows, kernel_columns))
    for irow in range(0, kernel_rows):
        for icol in range(0, kernel_columns):
            D = np.sqrt(((midr - irow) ** 2) + ((midc - icol) ** 2))
            k[irow, icol] = np.cos(D * np.pi / 2 / maxD)

    return k / np.sum(k.ravel())


def smoothc(mI: np.ndarray, Nr: int, Nc: int) -> np.ndarray:
    """

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
    """
    # Determine convolution kernel k
    k = get_smoothing_kernel(Nr, Nc)
    # Perform convolution
    out = np.rot90(convolve2d(np.rot90(mI, 2), np.rot90(k, 2), mode='same'), 2)
    return out[Nr:-Nr, Nc:-Nc]


def desmooth(pixels: np.ndarray, nx: int, ny: int) -> np.ndarray:
    smoothed_pixels = smooth2(pixels, nx, ny)
    desmoothed_pixels = pixels - smoothed_pixels
    return desmoothed_pixels


def smooth2(M: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """
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

    """
    S = np.concatenate((np.tile(M[0, :], (nx, 1)).transpose(),
                        M.transpose(),
                        np.tile(M[-1, :], (nx, 1)).transpose()), axis=1).transpose()

    T = np.concatenate((np.tile(S[:, 0], (ny, 1)).transpose(),
                        S,
                        np.tile(S[:, -1], (ny, 1)).transpose()), axis=1)

    return smoothc(T, nx, ny)


def circular_masking(image_array: np.ndarray) -> np.ndarray:
    mask = circular_mask(image_array.shape[0], image_array.shape[1], image_array.dtype)
    return image_array * mask


def normalise(image_array: np.ndarray) -> np.ndarray:
    """Performs normalisation of the matrix

    :param image_array: entry image
    :returns: normalised image
    """
    norm_image = (image_array - np.nanmean(image_array)) / np.nanstd(image_array)
    return norm_image


def gaussian_masking(image_array: np.ndarray, sigma: float) -> np.ndarray:
    """ Apply a gaussian mask to a matrix

    :param image_array: entry image
    :param sigma: standard deviation of the gaussian
    :returns: gaussian maked image
    """
    mask = gaussian_mask(image_array.shape[0], image_array.shape[1], sigma)
    return image_array * mask
