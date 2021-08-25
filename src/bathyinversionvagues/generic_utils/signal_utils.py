# -*- coding: utf-8 -*-
""" Module gathering several tools about one dimension signal

:created: 25/08/2021
:author: Romain Degoul
"""

import numpy as np


def find_period(signal: np.ndarray) -> float:
    """
    This fonction computes period of the signal by comuting the zeros of the signal
    The signal is supposed to be periodic and centered around zero
    :param signal: signal on which period is computed
    :return: period
    """
    sign = np.sign(signal)
    diff = np.diff(sign)
    zeros = np.where(diff != 0)[0]
    period = 2 * np.mean(np.diff(zeros))
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