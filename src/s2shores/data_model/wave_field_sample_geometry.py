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
from typing import Callable, List

import numpy as np


class WaveFieldSampleGeometry:
    """ This class encapsulates the geometric information defining a sample of a wave field:

    - its direction relative to some origin direction (image or geographical azimuth),
    - its wavelength, considering that a wave field is modeled by a periodic pattern

    This information is strictly local (thus the term sample) and contains only elements which are
    observable. For instance no information related to the dynamics or the bathymetry or anything
    else is contained herein.
    """

    def __init__(self) -> None:
        self._direction = np.nan
        self._wavelength = np.nan
        self._wavelength_change_observers: List[Callable] = []

    @property
    def direction(self) -> float:
        """ :returns: The propagation direction relative to the X axis (counterclockwise) (degrees)
        :raises ValueError: when the direction is not between -180° and +180° (inclusive)
        """
        return self._direction

    @direction.setter
    def direction(self, value: float) -> None:
        if value < -180. or value > 180.:
            raise ValueError('Direction must be between -180° and +180°')
        self._direction = value

    def _invert_direction(self) -> None:
        """ Invert the current direction by adding or subtracting 180°
        """
        if not np.isnan(self.direction):
            if self.direction < 0:
                self.direction += 180
            else:
                self.direction -= 180

    @property
    def direction_from_north(self) -> float:
        """ :returns: The direction relative to the North from which the wave field comes from,
        counted clockwise (degrees)"""
        return (270. - self._direction) % 360.

    @property
    def wavelength(self) -> float:
        """ :returns: The wave field wavelength (m)
        :raises ValueError: when the wavelength is not positive.
        """
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value: float) -> None:
        if value != self._wavelength:
            if value < 0:
                raise ValueError('Wavelength must be positive')
            self._wavelength = value
            for notify in self._wavelength_change_observers:
                notify()

    @property
    def wavenumber(self) -> float:
        """ :returns: The wave field wave number (m-1)"""
        return 1. / self._wavelength

    @wavenumber.setter
    def wavenumber(self, value: float) -> None:
        self.wavelength = 1. / value

    def register_wavelength_change(self, notify: Callable) -> None:
        """ Register the functions to be called whenever a change of the wavelength value occurs.

        :param notify: a function without argument which must be called when the wavelength value
                       is changed
        """
        self._wavelength_change_observers.append(notify)

    def __str__(self) -> str:
        result = f'Geometry:   direction: {self.direction}° '
        result += f'wavelength: {self.wavelength:5.2f} (m) wavenumber: {self.wavenumber:8.6f} (m-1)'
        return result
