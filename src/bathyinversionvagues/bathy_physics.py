# -*- coding: utf-8 -*-
""" Functions related to bathymetry physics.

:authors: erwinbergsma, gregoirethoumyre
:created: Mon Mar 23 2020
"""

# Imports
from typing import Tuple

import numpy as np


def funLinearC_k(nu: float, celerity: float,
                 precision: float = 0.0005, gravity: float = 9.81) -> float:
    # FIXME: What happens if celerity=0? infinite loop ?
    k = 2 * np.pi * nu
    w = celerity * k
    estimated_depth = celerity ** 2 / gravity

    nb_iter = 0
    previous_depth = np.Infinity
    while (abs(previous_depth - estimated_depth) > precision):
        nb_iter += 1
        previous_depth = estimated_depth
        dispe = w ** 2 - (gravity * k * np.tanh(k * previous_depth))
        fdispe = -gravity * (k ** 2) / (np.cosh(k * previous_depth) ** 2)
        estimated_depth = previous_depth - (dispe / fdispe)
    return estimated_depth


def phi_limits(wave_numbers: np.ndarray, delta_t: float,
               min_depth: float, gravity: float) -> Tuple[np.ndarray, np.ndarray]:

    delta_phi = 2 * np.pi * delta_t
    squeezed_wave_numbers = wave_numbers.squeeze()
    # shallow water limits:
    min_celerity = np.sqrt(gravity * min_depth)
    phi_min = delta_phi * min_celerity * squeezed_wave_numbers

    # deep water limits:
    phi_max = delta_phi / get_period_offshore(squeezed_wave_numbers, gravity)

    return phi_min, phi_max


def get_period_offshore(wave_numbers: np.ndarray, gravity: float) -> np.ndarray:
    return np.sqrt(2. * np.pi / (gravity * wave_numbers))
