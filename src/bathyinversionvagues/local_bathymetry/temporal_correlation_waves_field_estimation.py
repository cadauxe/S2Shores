# -*- coding: utf-8 -*-
""" Class handling the information describing a waves field sample..

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 10 sep 2021
"""
import numpy as np

from .waves_field_estimation import WavesFieldEstimation


class TemporalCorrelationWavesFieldEstimation(WavesFieldEstimation):
    """ This class encapsulates the information estimated in a waves field sample by a
    TemporalCorrelationBathyEstimator.
    """
    
    def __init__(self, gravity: float, depth_estimation_method: str) -> None:

        super().__init__(gravity, depth_estimation_method)
        self._travelled_distance = np.nan


    @property
    def travelled_distance(self) -> float:
        """ :returns: the travelled distance during temporal lag """
        return self._travelled_distance

    @travelled_distance.setter
    def travelled_distance(self, value: float) -> None:
        self._travelled_distance = value
