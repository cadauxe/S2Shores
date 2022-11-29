# -*- coding: utf-8 -*-
""" Class handling the information describing a wave field sample..

:author: Grégoire Thoumyre
:organization: CNES/LEGOS
:copyright: 2021 CNES/LEGOS. All rights reserved.
:license: see LICENSE file
:created: 20 sep 2021
"""
from typing import Tuple

import numpy as np

from ..data_model.bathymetry_sample_estimation import \
    BathymetrySampleEstimation


class SpatialCorrelationBathyEstimation(BathymetrySampleEstimation):
    """ This class encapsulates the information estimated in a wave field sample by a
    SpatialCorrelationBathyEstimator.

    It defines the estimation attributes specific to this estimator.
    """

    def __init__(self, gravity: float, depth_estimation_method: str,
                 period_range: Tuple[float, float], linearity_range: Tuple[float, float],
                 shallow_water_limit: float) -> None:
        """ Constructor

        :param gravity: the acceleration of gravity to use (m.s-2)
        :param shallow_water_limit: the depth limit between intermediate and shallow water (m)
        :param depth_estimation_method: the name of the depth estimation method to use
        :param period_range: minimum and maximum values allowed for the period
        :param linearity_range: minimum and maximum values allowed for the linearity indicator
        :raises NotImplementedError: when the depth estimation method is unsupported
        """

        super().__init__(gravity, depth_estimation_method, period_range, linearity_range,
                         shallow_water_limit)

        self._energy = np.nan

    @property
    def delta_celerity(self) -> float:
        # FIXME: define this quantity
        """ :returns: TBD """
        return np.nan

    @property
    def energy(self) -> float:
        """ :returns: the energy of the wave field """
        return self._energy

    @energy.setter
    def energy(self, value: float) -> None:
        self._energy = value

    @property
    def energy_ratio(self) -> float:
        """ :returns: The ratio of energy relative to the max peak """
        return (self.relative_period ** 2) * self.energy

    def __str__(self) -> str:
        result = super().__str__()
        result += f'\n    energy: {self.energy:5.2f} (???)'
        result += f'  energy ratio: {self.energy_ratio:5.2f} '
        return result
