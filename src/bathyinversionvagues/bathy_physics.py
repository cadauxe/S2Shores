# -*- coding: utf-8 -*-
""" Functions related to bathymetry physics.

:authors: erwinbergsma, gregoirethoumyre
:created: Mon Mar 23 2020
"""
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


def period_low_depth(wavenumber: NdArrayOrFloat, min_depth: float,
                     gravity: float) -> NdArrayOrFloat:
    """ Computes the waves period limit in shallow water

    :param wavenumber: wavenumber(s) of the waves (1/m)
    :param min_depth: minimum depth limit (m)
    :param gravity: acceleration of the gravity (m/s2)
    :returns: the waves period limit in shallow water (s)
    """
    return 1. / (celerity_low_depth(min_depth, gravity) * wavenumber)


def celerity_low_depth(shallow_water_depth: float, gravity: float) -> float:
    """ Computes the celerity in shallow water

    :param shallow_water_depth: minimum depth limit (m)
    :param gravity: acceleration of the gravity (m/s2)
    :returns: the celerity in shallow water (m/s)
    """
    return np.sqrt(gravity * shallow_water_depth)


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

    :param period: period(s) of the waves (s)
    :param gravity: acceleration of the gravity (m/s2)
    :returns: the wavelength according to the linear dispersive relation (m)
    """
    return 1. / wavenumber_offshore(period, gravity)


def celerity_offshore(period: NdArrayOrFloat, gravity: float) -> NdArrayOrFloat:
    """ Computes the celerity from the period under the offshore hypothesis

    :param gravity: acceleration of the gravity (m/s2)
    :param period: period(s) of the waves (s).
    :returns: the celerity according to the linear dispersive relation (m.s-1)
    """
    return (gravity / 2. * np.pi) * period
