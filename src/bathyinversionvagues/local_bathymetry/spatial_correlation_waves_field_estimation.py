# -*- coding: utf-8 -*-
""" Class handling the information describing a waves field sample..

:author: Grégoire Thoumyre
:organization: CNES/LEGOS
:copyright: 2021 CNES/LEGOS. All rights reserved.
:license: see LICENSE file
:created: 20 sep 2021
"""
import numpy as np

from .waves_field_estimation import WavesFieldEstimation


class SpatialCorrelationWavesFieldEstimation(WavesFieldEstimation):
    """ This class encapsulates the information estimated in a waves field sample by a
    SpatialCorrelationBathyEstimator.

    It defines the estimation attributes specific to this estimator.
    """

    def __init__(self, gravity: float,
                 depth_estimation_method: str, depth_precision: float) -> None:
        """ Constructor

        :param gravity: the acceleration of gravity to use (m.s-2)
        :param depth_estimation_method: the name of the depth estimation method to use
        :param depth_precision: precision (in meters) to be used for depth estimation
        """
        super().__init__(gravity, depth_estimation_method, depth_precision)

        self._delta_time = np.nan
        self._correlation_signal = None  # TODO: set to a matrix of nan

    @property
    def delta_time(self) -> float:
        """ :returns: the time difference between the 2 images used for this estimation """
        return self._delta_time

    @delta_time.setter
    def delta_time(self, value: float) -> None:
        self._delta_time = value

    @property
    def correlation_signal(self) -> np.ndarray:
        """ :returns: the spatial correlation between the 2 radon transform images"""
        return self._correlation_signal

    @correlation_signal.setter
    def correlation_signal(self, value: np.ndarray) -> None:
        self._correlation_signal = value

    def __str__(self) -> str:
        result = WavesFieldEstimation.__str__(self)
        result += f'\ndelta time: {self.delta_time:5.2f} (s)'
        return result
