# -*- coding: utf-8 -*-
""" Module gathering several tools about one dimension signal

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 01/09/2021

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""

import numpy as np
from scipy.signal import fftconvolve


def cross_correlation(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """ Compute the correlation of each line of image1 with each line of image2
    This function is faster than using np.corrcoef which computes correlation between
    A&A,A&B,B&A,B&B

    :param image1 : matrix image1
    :param image2 : matrix image2
    :return: cross correlation matrix
    :raises ValueError : when cross_correlation can not be computed
    """
    # Rowwise mean of input arrays & subtract from input arrays themselves
    image1_c = image1 - image1.mean(1)[:, None]
    image2_c = image2 - image2.mean(1)[:, None]

    # Sum of squares across rows
    ss1 = (image1_c ** 2).sum(1)
    ss2 = (image2_c ** 2).sum(1)
    product_deviation = np.sqrt(np.dot(ss1[:, None], ss2[None]))
    if np.any(product_deviation == 0):
        raise ValueError(
            'Cross correlation can not be computed because of standard deviation of 0')

    # Finally get corr coeff
    return np.divide(np.dot(image1_c, image2_c.T), product_deviation)


def normxcorr2(template: np.ndarray, image: np.ndarray, mode: str = 'full') -> np.ndarray:
    """ Compute the correlation of a template with an image

    :param template : matrix template
    :param image : matrix image
    :param mode : options among "full", "valid", "same"
    :return: correlation matrix
    """
    ########################################################################################
    # Author: Ujash Joshi, University of Toronto, 2017                                     #
    # Based on Octave implementation by: Benjamin Eltzner, 2014 <b.eltzner@gmx.de>         #
    # Octave/Matlab normxcorr2 implementation in python 3.5                                #
    # Details:                                                                             #
    # Normalized cross-correlation. Similiar results upto 3 significant digits.            #
    # https://github.com/Sabrewarrior/normxcorr2-python/master/norxcorr2.py                #
    # http://lordsabre.blogspot.ca/2017/09/matlab-normxcorr2-implemented-in-python.html    #
    ########################################################################################
    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if
                 template.shape[i] > image.shape[i]]) > 0:
        print('normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.')

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


def normalized_cross_correlation(template: np.ndarray, comparison: np.ndarray,
                                 correlation_mode: str) -> np.ndarray:
    """ :returns: the cross correlation 1D array between a template and a comparison sinogram
    """
    comparison = (comparison - np.mean(comparison)) / np.std(comparison)
    template = (template - np.mean(template)) / np.std(template)

    norm_cross_corr = np.correlate(template, comparison, correlation_mode)
    size_sinogram = len(template)
    size_crosscorr = len(norm_cross_corr)
    ind_min = (size_crosscorr - size_sinogram) // 2
    ind_max = (size_crosscorr + size_sinogram) // 2
    norm_cross_corr_out = norm_cross_corr[ind_min:ind_max] / size_sinogram

    return norm_cross_corr_out
