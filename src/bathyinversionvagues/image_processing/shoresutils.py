# -*- coding: utf-8 -*-
"""
Created on Wed Feb 3 10:12:00 2021

Module containing all functions common to waves and bathy estimation methods

@author: erwinbergsma
         gregoirethoumyre
         degoulromain

"""
from scipy.signal import detrend

import numpy as np


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



