# -*- coding: utf-8 -*-
""" Functions related to bathymetry physics.

:authors: erwinbergsma, gregoirethoumyre
:created: Mon Mar 23 2020
"""

# Imports
import math
from typing import Tuple, Union, Optional

import numpy as np


NdArrayOrFloat = Union[np.ndarray, float]


def depth_from_dispersion(wavenumber: float, celerity: float, gravity: float) -> float:
    angular_wavenumber = 2 * np.pi * wavenumber
    factor = celerity**2 * angular_wavenumber / gravity
    if abs(factor) > 1.:
        depth = np.Infinity
    else:
        depth = math.atanh(factor) / angular_wavenumber
    return depth


def phi_limits(wavenumber: NdArrayOrFloat, delta_t: float,
               min_depth: float, gravity: float) -> Tuple[NdArrayOrFloat, NdArrayOrFloat]:

    delta_phi = 2 * np.pi * delta_t

    celerity_min, celerity_max = celerity_limits(min_depth, gravity, wavenumber=wavenumber)
    phi_min = delta_phi * celerity_min * wavenumber
    phi_max = delta_phi * celerity_max * wavenumber

    return phi_min, phi_max


def celerity_limits(min_depth: float, gravity: float,
                    period: Optional[NdArrayOrFloat] = None,
                    wavenumber: Optional[NdArrayOrFloat] = None) -> Tuple[float, NdArrayOrFloat]:
    """ Computes the celerity limits either from the period or from the wavenumber

    :param min_depth: the minimum depth under which celerity does not depend on wavenumber (m)
    :param gravity: acceleration of the gravity (m/s2)
    :param period: period(s) of the waves (s)
    :param wavenumber: wavenumber(s) of the waves (1/m)
    :returns: the minimum and maximum celerities for the requested period(s) or wavenumber(s)
    """
    # celerity is minimal in shallow water an does not depend on the period nor on the wavenumber:
    celerity_min = np.sqrt(gravity * min_depth)

    # celerity is maximal in deep water depending on the period or wavenumber:
    celerity_max = celerity_offshore(gravity, period=period, wavenumber=wavenumber)
    return celerity_min, celerity_max


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

    :param period: period(s) of the waves (s)
    :param gravity: acceleration of the gravity (m/s2)
    :returns: the wavelength according to the linear dispersive relation (m)
    """
    return 1. / wavenumber_offshore(period, gravity)


def celerity_offshore(gravity: float,
                      period: Optional[NdArrayOrFloat] = None,
                      wavenumber: Optional[NdArrayOrFloat] = None,) -> NdArrayOrFloat:
    """ Computes the offshore celerity either from the period or from the wavenumber.
    period and wavenumber are mutually exclusive.

    :param gravity: acceleration of the gravity (m/s2)
    :param period: period(s) of the waves (s).
    :param wavenumber: wavenumber(s) of the waves (m-1)
    :returns: the celerity according to the linear dispersive relation (m.s-1)
    :raises ValueError: when period and wevenumber are both specified or when none are specified.
    """
    if period is None:
        if wavenumber is not None:
            return np.sqrt(gravity / (2. * np.pi * wavenumber))
        raise ValueError('pediod or wavenumber must be specified')
    else:
        if wavenumber is None:
            return (gravity / 2. * np.pi) * period
        raise ValueError('pediod and wavenumber cannot be specified at the same time')


def linearity_indicator(
        wavelength: NdArrayOrFloat, celerity: NdArrayOrFloat, gravity: float) -> NdArrayOrFloat:
    """ Computes a linearity indicator of the depth estimation using the linear dispersive relation

    :param wavelength: wavelength of the waves (m)
    :param celerity: the celerity of the waves field (m.s-1)
    :param gravity: acceleration of the gravity (m/s2)
    :returns: an indicator of the linearity between celerity and wavelength (unitless - [0, 1])
    """
    return 2 * np.pi * (celerity ** 2) / (gravity * wavelength)
