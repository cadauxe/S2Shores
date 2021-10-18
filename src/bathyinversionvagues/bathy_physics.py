# -*- coding: utf-8 -*-
""" Functions related to bathymetry physics.

:authors: erwinbergsma, gregoirethoumyre
:created: Mon Mar 23 2020
"""

# Imports
from typing import Tuple

import numpy as np


def funLinearC_k(wavenumber: float, celerity: float, precision: float, gravity: float) -> float:
    # FIXME: What happens if celerity=0? infinite loop ?
    k = 2 * np.pi * wavenumber
    w = celerity * k
    estimated_depth = celerity ** 2 / gravity

    previous_depth = np.Infinity
    while abs(previous_depth - estimated_depth) > precision:
        previous_depth = estimated_depth
        dispe = w ** 2 - (gravity * k * np.tanh(k * previous_depth))
        fdispe = -gravity * (k ** 2) / (np.cosh(k * previous_depth) ** 2)
        estimated_depth = previous_depth - (dispe / fdispe)
    return estimated_depth


def phi_limits(wave_numbers: np.ndarray, delta_t: float,
               min_depth: float, gravity: float) -> Tuple[np.ndarray, np.ndarray]:

    delta_phi = 2 * np.pi * abs(delta_t)
    squeezed_wave_numbers = wave_numbers.squeeze()
    # shallow water limits:
    min_celerity = np.sqrt(gravity * min_depth)
    phi_min = delta_phi * min_celerity * squeezed_wave_numbers

    # deep water limits:
    phi_max = delta_phi / period_offshore(squeezed_wave_numbers, gravity)

    return phi_min, phi_max


def period_offshore(wave_number: np.ndarray, gravity: float) -> np.ndarray:
    """ Computes the period from the wavenumber under the offshore hypothesis

    :param wave_number: wavenumber of the waves (1/m)
    :param gravity: acceleration of the gravity (m/s2)
    :returns: the period according to the linear dispersive relation (s)
    """
    return np.sqrt(2. * np.pi / (gravity * wave_number))


def wavenumber_offshore(period: np.ndarray, gravity: float) -> np.ndarray:
    """ Computes the wavenumber from the period under the offshore hypothesis

    :param period: period of the waves (s)
    :param gravity: acceleration of the gravity (m/s2)
    :returns: the wavenumber according to the linear dispersive relation (1/m)
    """
    return 2. * np.pi / (gravity * (period)**2)

def wavenumber_dual_period(period1: np.ndarray, period2: np.ndarray, gravity: float) -> np.ndarray:
    """ Computes the wavenumber from two different periods under the offshore hypothesis

        :param period1: first period (s)
        :param period2: second period (s)
        :param gravity: acceleration of the gravity (m/s2)
        :returns: the wavenumber according to the linear dispersive relation (1/m)
        """
    return 2. * np.pi / (gravity * period1 * period2)
