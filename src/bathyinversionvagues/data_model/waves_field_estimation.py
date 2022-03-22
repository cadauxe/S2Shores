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
        if np.isnan(value) or value == 0:
            period = np.nan
            value = np.nan
        else:
            period = self.delta_time * (2 * np.pi / value)
            if period < 0:
                self.invert_direction()
                value = -value
                period = -period
        self.period = period
        self._delta_phase = value

    def __str__(self) -> str:
        result = WavesFieldSampleBathymetry.__str__(self)
        result += f'\nEstimation: delta time: {self.delta_time:5.2f} (s)'
        return result
