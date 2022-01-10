# -*- coding: utf-8 -*-
""" Module gathering several tools about one dimension signal

:created: 25/08/2021
:author: Romain Degoul
"""

from typing import Tuple

from scipy.signal import find_peaks

import numpy as np


def find_period_from_zeros(signal: np.ndarray, min_period: int) -> Tuple[float, np.ndarray]:
    """ This function computes period of the signal by computing the zeros of the signal
    The signal is supposed to be periodic and centered around zero

    :param signal: signal on which period is computed
    :param min_period: minimal period between two detected points
    :return: (period,zeros used to compute period)
    """
    sign = np.sign(signal)
    diff = np.diff(sign)
    zeros = np.where(diff != 0)[0]
    demiperiods = np.diff(zeros)
    cond = demiperiods > (min_period / 2)
    demiperiods = demiperiods[cond]
    period = 2 * float(np.mean(demiperiods))
    return period, np.concatenate((np.array([zeros[0]]), zeros[1:][cond]))


def find_period_from_peaks(signal: np.ndarray, min_period: int) -> Tuple[float, np.ndarray]:
    """This function computes period of the signal by computing the peaks of the signal
    The signal is supposed to be periodic

    :param signal: signal on which period is computed
    :param min_period: minimal period between two detected points
    :return: (period,peaks indices used to compute period)
    """
    arg_peaks_max, _ = find_peaks(signal, distance=min_period)
    period = float(np.mean(np.diff(arg_peaks_max)))
    return period, arg_peaks_max


def find_dephasing(signal: np.ndarray, period: float) -> Tuple[float, np.ndarray]:
    """ This function computes dephasing of the signal
    The dephasing corresponds to the distance between the center of the signal and the position of
    the maximum

    :param signal: signal on which dephasing is computed
    :param period: period of the signal
    :return: (dephasing,signal selected on one period)
    """
    size_sinogram = len(signal)
    left_limit = max(int(size_sinogram / 2 - period / 2), 0)
    right_limit = min(int(size_sinogram / 2 + period / 2), size_sinogram)
    signal_period = signal[left_limit:right_limit]
    argmax = np.argmax(signal_period)
    dephasing = np.abs(argmax - period / 2)
    return dephasing, signal_period


def get_unity_roots(frequencies: np.ndarray, number_of_roots: int) -> np.ndarray:
    """ Compute complex roots of the unity for some frequencies

    :param frequencies: 1D array of normalized frequencies where roots are needed
    :param number_of_roots: Number of unity roots to compute, starting from 0
    :returns: number_of_roots complex roots of the unity corresponding to fr frequencies
    """
    n = np.arange(number_of_roots)
    frequencies = np.expand_dims(frequencies, axis=1)
    unity_roots = np.exp(-2j * np.pi * frequencies * n)
    return unity_roots
