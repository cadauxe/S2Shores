# -*- coding: utf-8 -*-
""" Module gathering all image filters which can be applied on a 2D numpy array

:author: Romain Degoul
:organization: CNES
:created: 26 août 2021
"""

import numpy as np
from scipy.signal import medfilt


def filter_mean(array, window):
    if len(array) < 2 * window:
        raise ValueError("time serie is too small compared to the window")
    else:
        padded_time_serie = np.concatenate((np.full(window, np.mean(array[:window])),
                                            array,
                                            np.full(window, np.mean(array[-(window + 1):]))))
        return np.convolve(padded_time_serie, np.ones(2 * window + 1) / (2 * window + 1), 'valid')


def remove_median(array, kernel_ratio):
    kernel_size = round(len(array) * kernel_ratio)
    if (kernel_size % 2) == 0:
        kernel_size = kernel_size + 1
    return array - medfilt(array, kernel_size)


def normalize(array):
    return array / np.max(np.abs(array))
