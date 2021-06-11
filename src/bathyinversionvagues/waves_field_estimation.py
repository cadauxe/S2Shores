# -*- coding: utf-8 -*-
""" Class handling the information describing a waves field sample..

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 6 mars 2021
"""
import numpy as np

from .bathy_physics import get_period_offshore, funLinearC_k
from .waves_field_sample_dynamics import WavesFieldSampleDynamics


class WavesFieldEstimation(WavesFieldSampleDynamics):
    """ This class encapsulates the information estimating a waves field sample.

    It inherits from WavesFieldSampleDynamics and defines specific attributes related to the sample
    estimation based on physical bathymetry.
    """

    def __init__(self, delta_time: float, depth_precision: float, gravity: float) -> None:
        """ Constructor

        :param delta_time: the time difference between the 2 images used for the estimation
        :param depth_precision: precision (in meters) to be used for depth estimation
        :param gravity: the acceleration of gravity to use (m.s-2)
        """
        super().__init__()
        self._delta_time = delta_time
        self._delta_phase = np.nan
        self._delta_phase_ratio = np.nan
        self._energy_max = np.nan
        self._gravity = gravity
        self._depth_precision = depth_precision

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
            self.period = self._delta_time * (2 * np.pi / value)

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

    @property
    def ckg(self) -> float:
        """ :returns: ckg (unitless) """
        return self.wavenumber * 2 * np.pi * (self.celerity ** 2) / self._gravity

    @property
    def depth(self) -> float:
        """ :returns: The depth (m) """
        return funLinearC_k(self.wavenumber, self.celerity, self._depth_precision, self._gravity)

    @property
    def period_offshore(self) -> float:
        """ :returns: The offshore period (s) """
        return get_period_offshore(self.wavenumber, self._gravity)

    def __str__(self) -> str:
        result = WavesFieldSampleDynamics.__str__(self)
        result += f'\noffshore period: {self.period_offshore:5.2f} (s)'
        result += f'\ndelta phase: {self.delta_phase:5.2f} (rd)'
        result += f'  delta phase ratio: {self.delta_phase_ratio:5.2f} '
        result += f'\nenergy_max: {self.energy_max:5.2f} (???)'
        result += f'  energy_max ratio: {self.energy_ratio:5.2f} '
        result += f'\nckg: {self.ckg:5.2f} '
        result += f'  depth: {self.depth:5.2f} (m)'
        return result
