# -*- coding: utf-8 -*-
""" Class handling the information describing a waves field sample..

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 6 mars 2021
"""
import numpy as np

from .waves_field_sample_bathymetry import WavesFieldSampleBathymetry


class WavesFieldEstimation(WavesFieldSampleBathymetry):
    """ This class encapsulates the information estimating a waves field sample.

    It inherits from WavesFieldSampleDynamics and defines specific attributes related to the sample
    estimation based on physical bathymetry.
    """

    def __init__(self, gravity: float, depth_estimation_method: str) -> None:

        super().__init__(gravity, depth_estimation_method)
        self._delta_time = np.nan
        self._propagated_distance = np.nan
        self._delta_phase = np.nan

        self._updating_wavelength = False
        self.register_wavelength_change(self.wavelength_change_in_estimation)

        self._updating_period = False
        self.register_period_change(self.period_change_in_estimation)

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
    def propagated_distance(self) -> float:
        """ :returns: the propagated distance over time """
        return self._propagated_distance

    @propagated_distance.setter
    def propagated_distance(self, value: float) -> None:
        if value != self._propagated_distance:
            if np.isnan(value) or value == 0:
                value = np.nan
            else:
                if self.delta_time * value < 0:
                    # delta_time and propagated distance have opposite signs
                    self.invert_direction()
                    value = -value
            self._propagated_distance = value
            self._solve_shift_equations()

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
                    self.invert_direction()
                    value = -value
            self._delta_phase = value
            self._solve_shift_equations()

    def wavelength_change_in_estimation(self) -> None:
        """ When wavelength has changed (new value is ensured to be different from the previous one)
        either reset delta_phase and propagated_distance if both were set, or update one of them if
        the other is set.
        """
        if not self._updating_wavelength:
            if not np.isnan(self.delta_phase) and not np.isnan(self.propagated_distance):
                self._delta_phase = np.nan
                self._propagated_distance = np.nan
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
        propagated_distance_set = not np.isnan(self.propagated_distance)
        if wavelength_set and delta_phase_set and not propagated_distance_set:
            self._propagated_distance = self.wavelength * self.delta_phase / (2 * np.pi)
        elif wavelength_set and not delta_phase_set and propagated_distance_set:
            self._delta_phase = 2 * np.pi * self.propagated_distance / self.wavelength
        elif not wavelength_set and delta_phase_set and propagated_distance_set:
            self._updating_wavelength = True
            self.wavelength = 2 * np.pi * self.propagated_distance / self.delta_phase
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
        result = WavesFieldSampleBathymetry.__str__(self)
        result += f'\nEstimation: delta time: {self.delta_time:5.2f} (s)'
        return result
