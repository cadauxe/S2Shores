# -*- coding: utf-8 -*-
""" Module gathering all image filters which can be applied on a 2D numpy array

:author: Romain Degoul
:organization: CNES
:created: 26 août 2021
"""
from scipy.signal import medfilt

import numpy as np


def filter_mean(array: np.ndarray, window: int) -> np.ndarray:
    """ Performs mean filter on a signal

    :param array: entry signal
    :param window: size of the moving average window
    :returns: filtered array
    :raises ValueError: when the array is too small compared to window size
    """
    if len(array) < 2 * window:
        raise ValueError('array is too small compared to the window')
    padded_array = np.concatenate((np.full(window, np.mean(array[:window])),
                                   array,
                                   np.full(window, np.mean(array[-(window + 1):]))))
    return np.convolve(padded_array, np.ones(2 * window + 1) / (2 * window + 1), 'valid')


def remove_median(array: np.ndarray, kernel_ratio: float) -> np.ndarray:
    """ Performs median removal on a signal

    :param array: entry signal
    :param kernel_ratio: ratio size of the median kernel compared to the signal
    :returns: filtered array
    """
    kernel_size = round(len(array) * kernel_ratio)
    if (kernel_size % 2) == 0:
        kernel_size = kernel_size + 1
    return array - medfilt(array, kernel_size)
