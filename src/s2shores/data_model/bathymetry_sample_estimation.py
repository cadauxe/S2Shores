# -*- coding: utf-8 -*-
""" Class handling the information describing a wave field sample..

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 6 mars 2021

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
from typing import Tuple

from .bathymetry_sample_inversion import BathymetrySampleInversion
from .wave_field_sample_estimation import WaveFieldSampleEstimation


class BathymetrySampleEstimation(WaveFieldSampleEstimation, BathymetrySampleInversion):
    """ This class encapsulates the information estimating bathymetry on a sample.

    It inherits from WaveFieldSampleEstimation and BathymetrySampleInversion and defines specific
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

        WaveFieldSampleEstimation.__init__(self, period_range)
        BathymetrySampleInversion.__init__(
            self, gravity, shallow_water_limit, depth_estimation_method)

        self._linearity_range = linearity_range

    def __hash__(self) -> int:
        """ :returns: a hash code based on direction + wavelength + period
        """
        return hash((self.direction, self.wavelength, self.period))

    def is_physical(self) -> bool:
        """  Check if a bathymetry estimation on a sample satisfies physical constraints.

        :returns: True is the wave field is valid, False otherwise
        """
        # minimum and maximum values for the stroboscopic factor
        #   - minimum correspond to the stroboscopic factor for shallow water.
        #   - maximum correspond to the stroboscopic factor for offshore water.
        stroboscopic_factor_range = (self.stroboscopic_factor_low_depth,
                                     self.stroboscopic_factor_offshore)
        return (self.is_wave_field_valid(stroboscopic_factor_range) and
                self.is_linearity_inside(self._linearity_range))

    @property
    def stroboscopic_factor_low_depth(self) -> float:
        """ :returns: the stroboscopic factor relative to the period limit in shallow water
    """
        return self.delta_time / self.period_low_depth

    @property
    def stroboscopic_factor_offshore(self) -> float:
        """ :returns: the stroboscopic factor relative to the period offshore.
        """
        return self.delta_time / self.period_offshore

    @property
    def absolute_stroboscopic_factor_offshore(self) -> float:
        """ :returns: the stroboscopic factor relative to the period offshore.
        """
        return abs(self.stroboscopic_factor_offshore)

    def __str__(self) -> str:
        result = WaveFieldSampleEstimation.__str__(self)
        result += '\n' + BathymetrySampleInversion.__str__(self)
        result += '\nBathymetry Estimation: '
        result += f' stroboscopic factor low depth: {self.stroboscopic_factor_low_depth:5.3f} '
        result += f' stroboscopic factor offshore: {self.stroboscopic_factor_offshore:5.3f} '
        return result
