# -*- coding: utf-8 -*-
""" Class handling the information describing a wave field sample..

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 6 mars 2021

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
from typing import Tuple, cast

import numpy as np

from ..bathy_physics import (depth_from_dispersion, linearity_indicator, period_low_depth,
                             period_offshore, wavelength_offshore)
from .wave_field_sample_dynamics import WaveFieldSampleDynamics

KNOWN_DEPTH_ESTIMATION_METHODS = ['LINEAR']


class BathymetrySampleInversion(WaveFieldSampleDynamics):
    """ This class encapsulates the bathymetric inversion for a given sample.

    It inherits from WaveFieldSampleDynamics and defines specific attributes related to the
    bathymetry for that sample..
    """

    def __init__(self, gravity: float, shallow_water_limit: float,
                 depth_estimation_method: str) -> None:
        """ Constructor

        :param gravity: the acceleration of gravity to use (m.s-2)
        :param shallow_water_limit: the depth limit between intermediate and shallow water (m)
        :param depth_estimation_method: the name of the depth estimation method to use
        :raises NotImplementedError: when the depth estimation method is unsupported
        """
        if depth_estimation_method not in KNOWN_DEPTH_ESTIMATION_METHODS:
            msg = f'{depth_estimation_method} is not a supported depth estimation method.'
            msg += f' Must be one of {KNOWN_DEPTH_ESTIMATION_METHODS}'
            raise NotImplementedError(msg)

        super().__init__()

        self._gravity = gravity
        self._shallow_water_limit = shallow_water_limit
        self._depth_estimation_method = depth_estimation_method

    @property
    def depth(self) -> float:
        """ The estimated depth

        :returns: The depth (m)
        :raises AttributeError: when the depth estimation method is not supported
        """
        if self._depth_estimation_method == 'LINEAR':
            estimated_depth = depth_from_dispersion(self.wavenumber, self.celerity, self._gravity)
        else:
            msg = 'depth attribute undefined when depth estimation method is not supported'
            raise AttributeError(msg)
        return estimated_depth

    @property
    def linearity(self) -> float:
        """ :returns: a linearity indicator for depth estimation (unitless) """
        return linearity_indicator(self.wavelength, self.celerity, self._gravity)

    def is_linearity_inside(self, linearity_range: Tuple[float, float]) -> bool:
        """ Check if the linearity indicator is within a given range of values.

        :param linearity_range: minimum and maximum values allowed for the linearity indicator
        :returns: True if the linearity indicator is between the minimum and maximum values, False
                  otherwise
        """
        return (not np.isnan(self.linearity) and
                self.linearity >= linearity_range[0] and self.linearity <= linearity_range[1])

    @property
    def period_offshore(self) -> float:
        """ :returns: The offshore period (s) """
        return cast(float, period_offshore(self.wavenumber, self._gravity))

    @property
    def period_low_depth(self) -> float:
        """ :returns:  the period in shallow water (s)
        """
        return cast(float, period_low_depth(self.wavenumber,
                                            self._shallow_water_limit,
                                            self._gravity))

    @property
    def relative_period(self) -> float:
        """ :returns: the ratio of the period offshore over the period"""
        return self.period_offshore / self.period

    @property
    def wavelength_offshore(self) -> float:
        """ :returns: the wavelength offshore (s)"""
        return cast(float, wavelength_offshore(self.period, self._gravity))

    @property
    def relative_wavelength(self) -> float:
        """ :returns: the ratio of the wavelength offshore over the wavelength"""
        return self.wavelength_offshore / self.wavelength

    def __str__(self) -> str:
        result = f'Bathymetry inversion: depth: {self.depth:5.2f} (m) '
        result += f' gamma: {self.linearity:5.3f} '
        result += f' offshore period: {self.period_offshore:5.2f} (s) '
        result += f' shallow water period: {self.period_low_depth:5.2f} (s) '
        result += f' relative period: {self.relative_period:5.2f} '
        result += f' relative wavelength: {self.relative_wavelength:5.2f} '
        result += f' gravity: {self._gravity:5.3f} (s) '
        return result
