# -*- coding: utf-8 -*-
""" Functions related to bathymetry physics.

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2020 CNES. All rights reserved.
:license: see LICENSE file
:created: Mon Mar 23 2020

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
import math
from typing import Union

import numpy as np

NdArrayOrFloat = Union[np.ndarray, float]
SENSITIVITY_PRECISION = 0.0001


def linearity_indicator(wavelength: float, celerity: float, gravity: float) -> float:
    """ Computes a linearity indicator of the depth estimation using the linear dispersive relation

    :param wavelength: wavelength of the waves (m)
    :param celerity: the celerity of the waves (m.s-1)
    :param gravity: acceleration of the gravity (m/s2)
    :returns: an indicator of the linearity between celerity and wavelength (unitless, positive)
    """
    return 2 * np.pi * (celerity ** 2) / (gravity * wavelength)


def sensitivity_indicator(wavelength: float, celerity: float, gravity: float) -> float:
    """ Computes a sensitivity indicator of the depth estimation using the linear dispersive
    relation

    :param wavelength: wavelength of the waves (m)
    :param celerity: the celerity of the waves (m.s-1)
    :param gravity: acceleration of the gravity (m/s2)
    :returns:  sensitivity (1/slope of the linear dispersion relation)
      Sensitivity as a function of the derivative of the linear dispersion relation
    """

    # c2kg = gamma, we retrieve gamma via the linearity_indicator:
    gamma = linearity_indicator(wavelength, celerity, gravity)

    # Derivative of d/dGAMMA( atanh (GAMMA) ):
    # d/dGAMMA( GAMMA *  atanh (GAMMA) )  =  d/dGAMMA( GAMMA/2 ln( 1+GAMMA / 1-GAMMA ) )
    # derivative of that can be analitically be derived to
    # d/dGAMMA = atanh(GAMMA) + GAMMA / (1 - GAMMA^2)
    #
    # and sensitivity (1/mu):
    # mu = d/dGAMMA / tanh(GAMMA)
    # and the sensitivity is then 1 / mu:
    if abs(gamma) >= 1.:
        sensitivity = SENSITIVITY_PRECISION
    else:
        sensitivity = (1 / ((np.arctanh(gamma) + (gamma / (1 - (gamma**2)))) / np.tanh(gamma)))
    return sensitivity


def depth_from_dispersion(wavenumber: float, celerity: float, gravity: float) -> float:
    """ Estimate depth using the linear dispersive relation

    :param wavenumber: wavenumber of the waves (m-1)
    :param celerity: the celerity of the waves (m.s-1)
    :param gravity: acceleration of the gravity (m/s2)
    :returns: the depth according to the linear dispersion relation, or np.inf if the
              linearirty indicator is greater than 1.
    """
    factor = linearity_indicator(1. / wavenumber, celerity, gravity)
    if abs(factor) > 1.:
        depth = np.inf
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
