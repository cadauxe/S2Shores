# -*- coding: utf-8 -*-
""" Class handling the information describing a waves field sample.

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 10 sep 2021
"""
import numpy as np

from .waves_field_estimation import WavesFieldEstimation


class SpatialDFTWavesFieldEstimation(WavesFieldEstimation):
    """ This class encapsulates the information estimated in a waves field sample by a
    SpatialDFTBathyEstimator.

    It defines the estimation attributes specific to this estimator.
    """

    def __init__(self, gravity: float, depth_estimation_method: str) -> None:

        super().__init__(gravity, depth_estimation_method)

        self._delta_time = np.nan
        self._delta_phase = np.nan
        self._delta_phase_ratio = np.nan
        self._energy_max = np.nan

    @property
    def delta_celerity(self) -> float:
        # FIXME: define this quantity
        """ :returns: TBD """
        return np.nan

    @property
    def delta_phase(self) -> float:
        """ :returns: the measured phase difference between both observations (rd) """
        return self._delta_phase

    @delta_phase.setter
    def delta_phase(self, value: float) -> None:
        self._delta_phase = value
        if np.isnan(value) or value == 0:
            self.period = np.nan
        else:
            if (self._delta_time < 0. and value > 0.) or (self._delta_time > 0. and value < 0.):
                self.direction += 180
                if value < 0:
                    self._delta_phase = -self._delta_phase
            self.period = abs(self._delta_time * (2 * np.pi / self._delta_phase))

    @property
    def delta_time(self) -> float:
        """ :returns: the time difference between the 2 images used for this estimation """
        return self._delta_time

    @delta_time.setter
    def delta_time(self, value: float) -> None:
        self._delta_time = value

    @property
    def delta_phase_ratio(self) -> float:
        # FIXME: define this quantity
        """ :returns: the ratio of the phase difference compared to TBD """
        return self._delta_phase_ratio

    @delta_phase_ratio.setter
    def delta_phase_ratio(self, value: float) -> None:
        self._delta_phase_ratio = value

    @property
    def energy_max(self) -> float:
        # FIXME: define this quantity
        """ :returns: TBD """
        return self._energy_max

    @energy_max.setter
    def energy_max(self, value: float) -> None:
        self._energy_max = value

    @property
    def energy_ratio(self) -> float:
        """ :returns: The ratio of energy relative to the max peak """
        return (self.delta_phase_ratio ** 2) * self.energy_max

    def __str__(self) -> str:
        result = WavesFieldEstimation.__str__(self)
        result += f'\ndelta time: {self.delta_time:5.2f} (s)'
        result += f'\ndelta phase: {self.delta_phase:5.2f} (rd)'
        result += f'  delta phase ratio: {self.delta_phase_ratio:5.2f} '
        result += f'\nenergy_max: {self.energy_max:5.2f} (???)'
        result += f'  energy_max ratio: {self.energy_ratio:5.2f} '
        return result
