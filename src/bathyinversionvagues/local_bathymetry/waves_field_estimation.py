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

KNOWN_DEPTH_ESTIMATION_METHODS = ['LINEAR']


class WavesFieldEstimation(WavesFieldSampleBathymetry):
    """ This class encapsulates the information estimating a waves field sample.

    It inherits from WavesFieldSampleDynamics and defines specific attributes related to the sample
    estimation based on physical bathymetry.
    """

    def __init__(self, delta_time: float, gravity: float,
                 depth_estimation_method: str, depth_precision: float) -> None:
        """ Constructor

        :param delta_time: the time difference between the 2 images used for the estimation
        :param gravity: the acceleration of gravity to use (m.s-2)
        :param depth_estimation_method: the name of the depth estimation method to use
        :param depth_precision: precision (in meters) to be used for depth estimation
        :raises NotImplementedError: when the depth estimation method is unsupported
        """
        super().__init__(gravity, depth_estimation_method, depth_precision)

        self._delta_time = delta_time
        self._delta_phase = np.nan
        self._delta_phase_ratio = np.nan
        self._energy_max = np.nan

    @property
    def delta_celerity(self) -> float:
        # FIXME: define this quantity
        """ :returns: ????????????????? """
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
    def delta_phase_ratio(self) -> float:
        """ :returns: the ratio of the phase difference compared to ???? """
        return self._delta_phase_ratio

    @delta_phase_ratio.setter
    def delta_phase_ratio(self, value: float) -> None:
        self._delta_phase_ratio = value

    @property
    def energy_max(self) -> float:
        """ :returns: ??? """
        return self._energy_max

    @energy_max.setter
    def energy_max(self, value: float) -> None:
        self._energy_max = value

    @property
    def energy_ratio(self) -> float:
        """ :returns: The ratio of energy relative to the max peak """
        return (self.delta_phase_ratio ** 2) * self.energy_max

    def __str__(self) -> str:
        result = WavesFieldSampleBathymetry.__str__(self)
        result += f'\ndelta phase: {self.delta_phase:5.2f} (rd)'
        result += f'  delta phase ratio: {self.delta_phase_ratio:5.2f} '
        result += f'\nenergy_max: {self.energy_max:5.2f} (???)'
        result += f'  energy_max ratio: {self.energy_ratio:5.2f} '
        return result
