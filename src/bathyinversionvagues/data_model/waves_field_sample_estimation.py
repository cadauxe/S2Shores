# -*- coding: utf-8 -*-
""" Class handling the information describing a waves field sample..

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 6 mars 2021
"""
from typing import Tuple
import numpy as np


from .waves_field_sample_dynamics import WavesFieldSampleDynamics


class WavesFieldSampleEstimation(WavesFieldSampleDynamics):
    """ This class encapsulates the information estimating a waves field sample.

    It inherits from WavesFieldSampleDynamics and defines specific attributes related to the sample
    estimation based on physical bathymetry.
    """

    def __init__(self, period_range: Tuple[float, float]) -> None:
        """ Encapsulates the information related to the estimation of a waves field.

        :param period_range: minimum and maximum values allowed for the period
        """
        WavesFieldSampleDynamics.__init__(self)
        self._delta_time = np.nan
        self._delta_position = np.nan
        self._delta_phase = np.nan
        self._period_range = period_range

        self._updating_wavelength = False
        self.register_wavelength_change(self.wavelength_change_in_estimation)

        self._updating_period = False
        self.register_period_change(self.period_change_in_estimation)

    def is_waves_field_valid(self, ambiguity_range: Tuple[float, float]) -> bool:
        """  Check if a waves field estimation satisfies physical constraints.

        :param ambiguity_range: the minimum and maximum values allowed for the ambiguity
        :returns: True is the waves field is valid, False otherwise
        """
        return (self.is_period_inside(self._period_range) and
                self.is_ambiguity_inside(ambiguity_range))

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
    def ambiguity(self) -> float:
        """ :returns: the ratio of delta_time over the waves period. When its absolute value is
                      greater than 1, there is an  ambiguity in detecting the waves.
        """
        return self.delta_time / self.period

    @property
    def absolute_ambiguity(self) -> float:
        """ :returns: the ambiguity as a positive value.
        """
        return abs(self.ambiguity)

    def is_ambiguity_inside(self, ambiguity_range: Tuple[float, float]) -> bool:
        """ Check if the ambiguity is inside a given range of values.

        :param ambiguity_range: the minimum and maximum values allowed for the ambiguity
        :returns: True if the ambiguity is between a minimum and a maximum values, False otherwise.
        """
        ambiguity_min, ambiguity_max = ambiguity_range
        if ambiguity_min > ambiguity_max:
            ambiguity_min, ambiguity_max = ambiguity_max, ambiguity_min
        return (not np.isnan(self.ambiguity) and
                (ambiguity_min < self.ambiguity) and
                (self.ambiguity < ambiguity_max))

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
                if self.delta_time * value < 0:  # delta_time and delta_phase have opposite signs
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
        result = WavesFieldSampleDynamics.__str__(self)
        result += f'\nWaves Field Estimation: \n  delta time: {self.delta_time:5.3f} (s)'
        result += f' ambiguity: {self.ambiguity:5.3f} (unitless)'
        result += f'\n  delta position: {self.delta_position:5.2f} (m)'
        result += f'  delta phase: {self.delta_phase:5.2f} (rd)'
        return result
