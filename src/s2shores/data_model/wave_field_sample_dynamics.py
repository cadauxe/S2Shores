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
from typing import Callable, List, Tuple

import numpy as np

from .wave_field_sample_geometry import WaveFieldSampleGeometry


class WaveFieldSampleDynamics(WaveFieldSampleGeometry):
    """ This class encapsulates the information related to the dynamics of a wave field sample.
    It inherits from WaveFieldSampleGeometry which describes the observed field geometry,
    and contains specific attributes related to the field dynamics:

    - its period
    - its celerity

    """

    def __init__(self) -> None:
        """ Encapsulates the information related to the dynamics of a wave field sample, namely
        the wave field period and its celerity.
        """

        super().__init__()
        self._period = np.nan
        self._celerity = np.nan
        self._period_change_observers: List[Callable] = []

        self.register_wavelength_change(self.wavelength_change_in_dynamics)

    @property
    def period(self) -> float:
        """ :returns: The wave field period (s), which was either externally provided or computed
                      from the wavelength and the celerity
        :raises ValueError: when the period is not positive.
        """
        return self._period

    @period.setter
    def period(self, value: float) -> None:
        if value != self._period:
            if value < 0.:
                raise ValueError('Period must be positive')
            self._period = value
            if not np.isnan(self.celerity) and not np.isnan(self.wavelength):
                self._celerity = np.nan
                self.wavelength = np.nan
            self._solve_movement_equation()
            for notify in self._period_change_observers:
                notify()

    def is_period_inside(self, period_range: Tuple[float, float]) -> bool:
        """ Check if the wave field period is inside some range of values.

        :param period_range: minimum and maximum values allowed for the period
        :returns: True if the period is between the minimum and maximum values, False otherwise
        """
        return (not np.isnan(self.period) and
                self.period >= period_range[0] and self.period <= period_range[1])

    @property
    def celerity(self) -> float:
        """ :returns: The wave field velocity (m/s), which was either externally provided or
                      computed from the wavelength and the period
        :raises ValueError: when the celerity is not positive.
        """
        return self._celerity

    @celerity.setter
    def celerity(self, value: float) -> None:
        if value != self.celerity:
            if value < 0:
                raise ValueError('Celerity must be positive')
            self._celerity = value
            if not np.isnan(self.period) and not np.isnan(self.wavelength):
                self._period = np.nan
                self.wavelength = np.nan
            self._solve_movement_equation()

    def register_period_change(self, notify: Callable) -> None:
        """ Register the functions to be called whenever a change of the period value occurs.

        :param notify: a function without argument which must be called when the period value
                       is changed
        """
        self._period_change_observers.append(notify)

    def wavelength_change_in_dynamics(self) -> None:
        """ When wavelength has changed (new value is ensured to be different from the previous one)
        either reset period and celerity if both were set, or update one of them if the other is set
        """
        if not np.isnan(self.period) and not np.isnan(self.celerity):
            self._period = np.nan
            self._celerity = np.nan
        self._solve_movement_equation()

    def _solve_movement_equation(self) -> None:
        """ Solves the movement equation ( L=c*T ) when exactly one of the 3 variables is not set.
        In other cases does not change anything.
        """
        wavelength_set = not np.isnan(self.wavelength)
        period_set = not np.isnan(self.period)
        celerity_set = not np.isnan(self.celerity)
        if wavelength_set and period_set and not celerity_set:
            self._celerity = self.wavelength / self.period
        elif wavelength_set and not period_set and celerity_set:
            self._period = self.wavelength / self.celerity
        elif not wavelength_set and period_set and celerity_set:
            self.wavelength = self.celerity * self.period

    def __str__(self) -> str:
        result = WaveFieldSampleGeometry.__str__(self)
        result += f'\nDynamics:   period: {self.period:5.2f} (s)  '
        result += f'celerity: {self.celerity:5.2f} (m/s)'
        return result
