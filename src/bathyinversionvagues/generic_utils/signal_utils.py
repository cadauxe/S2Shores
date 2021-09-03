# -*- coding: utf-8 -*-
""" Module gathering several tools about one dimension signal

:created: 25/08/2021
:author: Romain Degoul
"""

import numpy as np


def find_period(signal: np.ndarray) -> float:
    """
    This function computes period of the signal by computing the zeros of the signal
    The signal is supposed to be periodic and centered around zero
    :param signal: signal on which period is computed
    :return: period
    """
    sign = np.sign(signal)
    diff = np.diff(sign)
    zeros = np.where(diff != 0)[0]
    period = 2 * float(np.mean(np.diff(zeros)))
    return period


def find_dephasing(signal: np.ndarray, period: float) -> float:
    """
    This function computes dephasing of the signal
    The dephasing corresponds to the distance between the center of the signal and the position of
    the maximum
    :param signal: signal on which dephasing is computed
    :param period: period of the signal
    :return: dephasing
    """
    size_sinogram = len(signal)
    left_limit = max(int(size_sinogram / 2 - period / 2), 0)
    right_limit = min(int(size_sinogram / 2 + period / 2), size_sinogram)
    argmax = np.argmax(signal[left_limit:right_limit])
    dephasing = np.abs(argmax + left_limit - size_sinogram / 2)
    return dephasing

def get_unity_roots(signal: np.ndarray, number_of_roots: int) -> np.ndarray:
    """
    Compute complex roots of the unity for some frequencies
    :param signal: 1D array of normalized frequencies where roots are needed
    :param number_of_roots: Number of unity roots to compute, starting from 0
    :returns: number_of_roots complex roots of the unity corresponding to fr frequencies
    """
    n = np.arange(number_of_roots)
    return np.exp(-2j * np.pi * signal * n)


def DFT_fr(signal: np.ndarray, unity_roots: np.ndarray):
    """ Compute the discrete Fourier Transform of a 1D array

    :param np.ndarray x: 1D array containing the signal
    :param np.ndarray
    """
    # FIXME: used to interpolate spectrum, but seems incorrect. Use zero padding instead ?
    return np.dot(unity_roots, signal)
