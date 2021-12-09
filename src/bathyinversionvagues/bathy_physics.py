# -*- coding: utf-8 -*-
""" Functions related to bathymetry physics.

:authors: erwinbergsma, gregoirethoumyre
:created: Mon Mar 23 2020
"""

# Imports
from typing import Tuple, Union

import numpy as np

NdArrayOrFloat = Union[np.ndarray, float]


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


def phi_limits(wavenumbers: np.ndarray, delta_t: float,
               min_depth: float, gravity: float) -> Tuple[NdArrayOrFloat, NdArrayOrFloat]:

    delta_phi = 2 * np.pi * delta_t
    squeezed_wavenumbers = wavenumbers.squeeze()
    # shallow water limits:
    min_celerity = np.sqrt(gravity * min_depth)
    phi_min = delta_phi * min_celerity * squeezed_wavenumbers

    # deep water limits:
    phi_max = delta_phi / period_offshore(squeezed_wavenumbers, gravity)

    return phi_min, phi_max


def period_offshore(wavenumber: NdArrayOrFloat, gravity: float) -> NdArrayOrFloat:
    """ Computes the period from the wavenumber under the offshore hypothesis

    :param wavenumber: wavenumber of the waves (1/m)
    :param gravity: acceleration of the gravity (m/s2)
    :returns: the period according to the linear dispersive relation (s)
    """
    return np.sqrt(2. * np.pi / (gravity * wavenumber))


def wavenumber_offshore(period: NdArrayOrFloat, gravity: float) -> NdArrayOrFloat:
    """ Computes the wavenumber from the period under the offshore hypothesis

    :param period: period of the waves (s)
    :param gravity: acceleration of the gravity (m/s2)
    :returns: the wavenumber according to the linear dispersive relation (1/m)
    """
    return 2. * np.pi / (gravity * (period)**2)


def wavelength_offshore(period: NdArrayOrFloat, gravity: float) -> NdArrayOrFloat:
    """ Computes the wavelength from the period under the offshore hypothesis

    :param period: period of the waves (s)
    :param gravity: acceleration of the gravity (m/s2)
    :returns: the wavelength according to the linear dispersive relation (m)
    """
    return 1. / wavenumber_offshore(period, gravity)


def celerity_offshore(period: NdArrayOrFloat, gravity: float) -> NdArrayOrFloat:
    """ Computes the celerity max from the period max under the offshore hypothesis

    :param period: period of the waves (s)
    :param gravity: acceleration of the gravity (m/s2)
    :returns: the celerity according to the linear dispersive relation (m.s-1)
    """
    return (gravity / 2. * np.pi) * period


def linearity_indicator(
        wavelength: NdArrayOrFloat, celerity: NdArrayOrFloat, gravity: float) -> NdArrayOrFloat:
    """ Computes a linearity indicator of the depth estimation using the linear dispersive relation

    :param wavelength: wavelength of the waves (m)
    :param celerity: the celerity of the waves field (m.s-1)
    :param gravity: acceleration of the gravity (m/s2)
    :returns: an indicator of the linearity between celerity and wavelength (unitless - [0, 1])
    """
    return 2 * np.pi * (celerity ** 2) / (gravity * wavelength)
