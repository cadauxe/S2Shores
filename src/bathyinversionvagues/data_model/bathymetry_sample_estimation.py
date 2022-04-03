# -*- coding: utf-8 -*-
""" Class handling the information describing a waves field sample..

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 6 mars 2021
"""
from typing import Tuple
import numpy as np


from .bathymetry_sample_inversion import BathymetrySampleInversion
from .waves_field_sample_estimation import WavesFieldSampleEstimation


class BathymetrySampleEstimation(WavesFieldSampleEstimation, BathymetrySampleInversion):
    """ This class encapsulates the information estimating bathymetry on a sample.

    It inherits from WavesFieldSampleEstimation and BathymetrySampleInversion and defines specific
    attributes related to the sample bathymetry estimation.
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

        WavesFieldSampleEstimation.__init__(self, period_range)
        BathymetrySampleInversion.__init__(
            self, gravity, shallow_water_limit, depth_estimation_method)

        self._linearity_range = linearity_range

    def is_physical(self) -> bool:
        """  Check if a bathymetry estimationon a sample satisfies physical constraints.

        :returns: True is the waves field is valid, False otherwise
        """
        # minimum and maximum values for the ambiguity:
        #   - minimum correspond to the ambiguity for shallow water.
        #   - maximum correspond to the ambiguity for offshore water.
        ambiguity_range = (self.ambiguity_low_depth, self.ambiguity_offshore)
        return (self.is_waves_field_valid(ambiguity_range) and
                self.is_linearity_inside(self._linearity_range))

    # FIXME: delta_phase_ratio should be removed. It is equal to period_ratio which has a more
    # generic meaning.
    @property
    def delta_phase_ratio(self) -> float:
        """ :returns: the fraction of the maximum phase shift allowable in deep waters """
        return self.delta_phase / (2 * np.pi * self.ambiguity_offshore)

    @property
    def ambiguity_low_depth(self) -> float:
        """ :returns: the ambiguity relative to the period limit in shallow water
    """
        return self.delta_time / self.period_low_depth

    @property
    def ambiguity_offshore(self) -> float:
        """ :returns: the ambiguity relative to the period offshore.
        """
        return self.delta_time / self.period_offshore

    def __str__(self) -> str:
        result = WavesFieldSampleEstimation.__str__(self)
        result += '\n' + BathymetrySampleInversion.__str__(self)
        result += f'\nBathymetry Estimation:  delta phase ratio: {self.delta_phase_ratio:5.2f} '
        result += f' ambiguity low depth: {self.ambiguity_low_depth:5.3f} '
        result += f' ambiguity offshore: {self.ambiguity_offshore:5.3f} '
        return result
