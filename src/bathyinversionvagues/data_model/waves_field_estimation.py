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

    @property
    def delta_time(self) -> float:
        """ :returns: the time difference between the images used for this estimation """
        return self._delta_time

    @delta_time.setter
    def delta_time(self, value: float) -> None:
        self._delta_time = value

    @property
    def propagated_distance(self) -> float:
        """ :returns: the propagated distance over time """
        return self._propagated_distance

    @propagated_distance.setter
    def propagated_distance(self, value: float) -> None:
        # FIXME: ensure propagated distance positive ?
        self._propagated_distance = value
        self.celerity = abs(value / self.delta_time)  # Must be bositive in all cases

    @property
    def delta_phase(self) -> float:
        """ :returns: the measured phase difference between both observations (rd) """
        return self._delta_phase

    @delta_phase.setter
    def delta_phase(self, value: float) -> None:
        self._delta_phase = value
        if np.isnan(self._delta_phase) or self._delta_phase == 0:
            self.period = np.nan
        else:
            period = self.delta_time * (2 * np.pi / self._delta_phase)
            if period < 0.:
                # delta_phase and delta_time have opposite signs, nothing to correct.
                # period must be positive
                period = abs(period)
            else:
                # delta_phase and delta_time have same signs, propagation direction must be inverted
                if self.direction < 0:
                    self.direction += 180
                else:
                    self.direction -= 180
            self.period = period

    def __str__(self) -> str:
        result = WavesFieldSampleBathymetry.__str__(self)
        result += f'\nEstimation: delta time: {self.delta_time:5.2f} (s)'
        return result
