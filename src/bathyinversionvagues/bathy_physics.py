# -*- coding: utf-8 -*-
""" Functions related to bathymetry physics.

:authors: erwinbergsma, gregoirethoumyre
:created: Mon Mar 23 2020
"""

# Imports
import math
from typing import Union

import numpy as np


NdArrayOrFloat = Union[np.ndarray, float]


def linearity_indicator(wavelength: float, celerity: float, gravity: float) -> float:
    """ Computes a linearity indicator of the depth estimation using the linear dispersive relation

    :param wavelength: wavelength of the waves (m)
    :param celerity: the celerity of the waves field (m.s-1)
    :param gravity: acceleration of the gravity (m/s2)
    :returns: an indicator of the linearity between celerity and wavelength (unitless, positive)
    """
    return 2 * np.pi * (celerity ** 2) / (gravity * wavelength)


def depth_from_dispersion(wavenumber: float, celerity: float, gravity: float) -> float:
    factor = linearity_indicator(1. / wavenumber, celerity, gravity)
    if abs(factor) > 1.:
        depth = np.Infinity
    else:
        depth = math.atanh(factor) / (2 * np.pi * wavenumber)
    return depth


def time_sampling_factor_low_depth(wavenumber: NdArrayOrFloat, delta_t: float, min_depth: float,
                                   gravity: float) -> NdArrayOrFloat:
    """ Computes the time sampling factor relative to the period limit in shallow water

    :param wavenumber: wavenumber(s) of the waves (1/m)
    :param delta_t: acquisition times difference (s)
    :param min_depth: minimum depth limit (m)
    :param gravity: acceleration of the gravity (m/s2)
    :returns: the time sampling factor relative to the period limit in shallow water (unitless)
    """
    celerity_low_depth_limit = np.sqrt(gravity * min_depth)
    period_low_depth_limit = 1. / (celerity_low_depth_limit * wavenumber)
    return delta_t / period_low_depth_limit


def time_sampling_factor_offshore(wavenumber: NdArrayOrFloat, delta_t: float,
                                  gravity: float) -> NdArrayOrFloat:
    """ Computes the time sampling factor relative to the period offshore

    :param wavenumber: wavenumber(s) of the waves (1/m)
    :param delta_t: acquisition times difference (s)
    :param gravity: acceleration of the gravity (m/s2)
    :returns: the time sampling factor relative to the period offshore (unitless)
    """

    return delta_t / period_offshore(wavenumber, gravity)


def period_offshore(wavenumber: NdArrayOrFloat, gravity: float) -> NdArrayOrFloat:
    """ Computes the period from the wavenumber under the offshore hypothesis

    :param wavenumber: wavenumber(s) of the waves (1/m)
    :param gravity: acceleration of the gravity (m/s2)
    :returns: the period according to the linear dispersive relation (s)
    """
    return np.sqrt(2. * np.pi / (gravity * wavenumber))


def wavenumber_offshore(period: NdArrayOrFloat, gravity: float) -> NdArrayOrFloat:
    """ Computes the wavenumber from the period under the offshore hypothesis

    :param period: period(s) of the waves (s)
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
    """ Computes the celerity from the period under the offshore hypothesis

    :param period: period of the waves (s)
    :param gravity: acceleration of the gravity (m/s2)
    :returns: the celerity according to the linear dispersive relation (m.s-1)
    """
    return (gravity / 2. * np.pi) * period
