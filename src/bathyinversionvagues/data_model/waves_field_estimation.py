# -*- coding: utf-8 -*-
""" Class handling the information describing a waves field sample..

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 6 mars 2021
"""
from typing import cast, Tuple
import numpy as np

from ..bathy_physics import time_sampling_factor_offshore, time_sampling_factor_low_depth
from .waves_field_sample_bathymetry import WavesFieldSampleBathymetry
from .waves_field_sample_estimation import WavesFieldSampleEstimation


class WavesFieldEstimation(WavesFieldSampleEstimation, WavesFieldSampleBathymetry):
    """ This class encapsulates the information estimating a waves field sample.

    It inherits from WavesFieldSampleDynamics and defines specific attributes related to the sample
    estimation based on physical bathymetry.
    """

    def __init__(self, gravity: float, depth_estimation_method: str) -> None:

        WavesFieldSampleEstimation.__init__(self)
        WavesFieldSampleBathymetry.__init__(self, gravity, depth_estimation_method)

    def is_physical(self, period_range: Tuple[float, float], linearity_range: Tuple[float, float],
                    shallow_water_limit: float) -> bool:
        """  Check if a waves field estimation satisfies physical constraints.

        :param period_range: the minimum and maximum periods allowed for the waves (s)
        :param linearity_range: the minimum and maximum allowed for the waves linearity (unitless)
        :param shallow_water_limit: the depth limit between intermediate and shallow water (m)
        :returns: True is the waves field is valid, False otherwise
        """
        time_sampling_factor_min = cast(float,
                                        time_sampling_factor_low_depth(self.wavenumber,
                                                                       self.delta_time,
                                                                       shallow_water_limit,
                                                                       self._gravity))
        time_sampling_factor_max = cast(float,
                                        time_sampling_factor_offshore(self.wavenumber,
                                                                      self.delta_time,
                                                                      self._gravity))
        time_sampling_factor_range = (time_sampling_factor_min, time_sampling_factor_max)
        return (self.is_period_valid(period_range) and
                self.is_time_sampling_factor_valid(time_sampling_factor_range) and
                self.is_linearity_valid(linearity_range))

    @property
    def delta_phase_ratio(self) -> float:
        """ :returns: the fraction of the maximum phase shift allowable in deep waters """
        time_sampling_offshore = cast(float,
                                      time_sampling_factor_offshore(self.wavenumber,
                                                                    self.delta_time,
                                                                    self._gravity))
        return self.delta_phase / (2 * np.pi * time_sampling_offshore)

    def __str__(self) -> str:
        result = WavesFieldSampleEstimation.__str__(self)
        result += '\n' + WavesFieldSampleBathymetry.__str__(self)
        result += f'\nBathymetry Estimation:  delta phase ratio: {self.delta_phase_ratio:5.2f} '
        return result
