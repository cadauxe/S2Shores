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
from typing import Tuple

import numpy as np

from .wave_field_sample_dynamics import WaveFieldSampleDynamics


class WaveFieldSampleEstimation(WaveFieldSampleDynamics):
    """ This class encapsulates the information estimating a wave field sample.

    It inherits from WaveFieldSampleDynamics and defines specific attributes related to the sample
    estimation based on physical bathymetry.
    """

    def __init__(self, period_range: Tuple[float, float]) -> None:
        """ Encapsulates the information related to the estimation of a wave field.

        :param period_range: minimum and maximum values allowed for the period
        """
        WaveFieldSampleDynamics.__init__(self)
        self._delta_time = np.nan
        self._delta_position = np.nan
        self._delta_phase = np.nan
        self._period_range = period_range

        self._updating_wavelength = False
        self.register_wavelength_change(self.wavelength_change_in_estimation)

        self._updating_period = False
        self.register_period_change(self.period_change_in_estimation)

    def is_wave_field_valid(self, stroboscopic_factor_range: Tuple[float, float]) -> bool:
        """  Check if a wave field estimation satisfies physical constraints.

        :param stroboscopic_factor_range: the minimum and maximum values allowed for the
                                          stroboscopic factor
        :returns: True is the wave field is valid, False otherwise
        """
        return (self.is_period_inside(self._period_range) and
                self.is_stroboscopic_factor_inside(stroboscopic_factor_range))

    @property
    def delta_time(self) -> float:
        """ :returns: the time difference between the images used for this estimation """
        return self._delta_time

    @delta_time.setter
    def delta_time(self, value: float) -> None:
        if value != self._delta_time:
            self._delta_time = value
            self._solve_shift_equations()

    @property
    def stroboscopic_factor(self) -> float:
        """ :returns: the ratio of delta_time over the wave field period. When its value is
                      equal to multiples of 1/2 no wave movement can be perceived. when its
                      fractional part is lower than 1/2 the movement is perceived in the right
                      direction, whereas it is perceived as retrograde when the fractional part
                      is greater than 1/2.
        """
        return self.delta_time / self.period

    @property
    def absolute_stroboscopic_factor(self) -> float:
        """ :returns: the stroboscopic factor as a positive value.
        """
        return abs(self.stroboscopic_factor)

    def is_stroboscopic_factor_inside(self, stroboscopic_factor_range: Tuple[float, float]) -> bool:
        """ Check if the stroboscopic factor is inside a given range of values.

        :param stroboscopic_factor_range: the minimum and maximum values allowed for the factor
        :returns: True if the factor is between a minimum and a maximum values, False otherwise.
        """
        stroboscopic_factor_min, stroboscopic_factor_max = stroboscopic_factor_range
        if stroboscopic_factor_min > stroboscopic_factor_max:
            stroboscopic_factor_max, stroboscopic_factor_min = stroboscopic_factor_range
        return (not np.isnan(self.stroboscopic_factor) and
                (stroboscopic_factor_min < self.stroboscopic_factor) and
                (self.stroboscopic_factor < stroboscopic_factor_max))

    @property
    def delta_position(self) -> float:
        """ :returns: the propagated distance over time """
        return self._delta_position

    @delta_position.setter
    def delta_position(self, value: float) -> None:
        if value != self._delta_position:
            if np.isnan(value) or value == 0:
                value = np.nan
            else:
                if self.delta_time * value < 0:
                            # delta_time and propagated distance have opposite signs
                    self._invert_direction()
                    value = -value

            self._delta_position = value
            self._solve_shift_equations()

    @property
    def absolute_delta_position(self) -> float:
        """ :returns: the absolute value of the propagated distance over time """
        return abs(self._delta_position)

    @property
    def delta_phase(self) -> float:
        """ :returns: the measured phase difference between both observations (rd) """
        return self._delta_phase

    @delta_phase.setter
    def delta_phase(self, value: float) -> None:
        if value != self._delta_phase:
            if np.isnan(value) or value == 0:
                value = np.nan
            else:
                if self.delta_time * value < 0:
                    # delta_time and delta_phase have opposite signs
                    self._invert_direction()
                    value = -value

            self._delta_phase = value
            self._solve_shift_equations()

    @property
    def absolute_delta_phase(self) -> float:
        """ :returns: the absolute value of the phase difference between both observations (rd) """
        return abs(self._delta_phase)

    def wavelength_change_in_estimation(self) -> None:
        """ When wavelength has changed (new value is ensured to be different from the previous one)
        either reset delta_phase and delta_position if both were set, or update one of them if
        the other is set.
        """
        if not self._updating_wavelength:
            if not np.isnan(self.delta_phase) and not np.isnan(self.delta_position):
                self._delta_phase = np.nan
                self._delta_position = np.nan
        self._solve_shift_equations()

    def period_change_in_estimation(self) -> None:
        """ When period has changed (new value is ensured to be different from the previous one)
        either reset delta_phase and delta_time if both were set, or update one of them if
        the other is set.
        """
        if not self._updating_period:
            if not np.isnan(self.delta_phase) and not np.isnan(self.delta_time):
                self._delta_phase = np.nan
                self._delta_time = np.nan
        self._solve_shift_equations()

    def _solve_shift_equations(self) -> None:
        """ Solves the shift equations involving spatial and temporal quantities
        """
        self._solve_spatial_shift_equation()
        self._solve_temporal_shift_equation()
        # Solve spatial dephasing equation again in case delta_phase has been set through temporal
        # dephasing equation.
        self._solve_spatial_shift_equation()

    def _solve_spatial_shift_equation(self) -> None:
        """ Solves the shift equation involving spatial quantities ( L*dPhi = 2*Pi*dX ) when
        exactly one of the 3 variables is not set. In other cases does not change anything.
        """
        delta_phase_set = not np.isnan(self.delta_phase)
        wavelength_set = not np.isnan(self.wavelength)
        delta_position_set = not np.isnan(self.delta_position)
        if wavelength_set and delta_phase_set and not delta_position_set:
            self._delta_position = self.wavelength * self.delta_phase / (2 * np.pi)
        elif wavelength_set and not delta_phase_set and delta_position_set:
            self._delta_phase = 2 * np.pi * self.delta_position / self.wavelength
        elif not wavelength_set and delta_phase_set and delta_position_set:
            self._updating_wavelength = True
            self.wavelength = 2 * np.pi * self.delta_position / self.delta_phase
            self._updating_wavelength = False

    def _solve_temporal_shift_equation(self) -> None:
        """ Solves the shift equation involving temporal quantities ( T*dPhi = 2*Pi*dT ) when
        exactly one of the 3 variables is not set. In other cases does not change anything.
        """
        delta_phase_set = not np.isnan(self.delta_phase)
        delta_time_set = not np.isnan(self.delta_time)
        period_set = not np.isnan(self.period)
        if delta_time_set and delta_phase_set and not period_set:
            self._updating_period = True
            self.period = 2 * np.pi * self.delta_time / self.delta_phase
            self._updating_period = False
        elif delta_time_set and not delta_phase_set and period_set:
            self._delta_phase = 2 * np.pi * self.delta_time / self.period
        elif not delta_time_set and delta_phase_set and period_set:
            self._delta_time = self.period * self.delta_phase / (2 * np.pi)

    def __str__(self) -> str:
        result = WaveFieldSampleDynamics.__str__(self)
        result += f'\nWave Field Estimation: \n  delta time: {self.delta_time:5.3f} (s)'
        result += f' stroboscopic factor: {self.stroboscopic_factor:5.3f} (unitless)'
        result += f'\n  delta position: {self.delta_position:5.2f} (m)'
        result += f'  delta phase: {self.delta_phase:5.2f} (rd)'
        return result
