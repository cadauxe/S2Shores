# -*- coding: utf-8 -*-
""" Definition of the DirectionsQuantizer class

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 12 Oct 2021

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
from typing import Optional, Union, cast  # @NoMove

import numpy as np

DEFAULT_QUANTIZATION_STEP = .1


class DirectionsQuantizer:
    def __init__(self, quantization_step: Optional[float] = None) -> None:
        """ Constructor

        :param quantization_step: the step to use for quantizing direction angles.
                                  Direction quantization is such that the 0 degree
                                  direction is used as the origin, and any direction angle is
                                  transformed to the nearest quantized angle for indexing purposes
        """
        self._quantization_step = \
            DEFAULT_QUANTIZATION_STEP if quantization_step is None else quantization_step

    @property
    def quantization_step(self) -> float:
        """ :returns: the step used to quantize directions """
        return self._quantization_step

    def quantize(self, direction: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(direction, float):
            return self.quantize_float(direction)
        # Firstly, normalize direction between -180 and +180 degrees
        normalized_direction = self.normalize(direction)

        index_direction = np.around(normalized_direction / self._quantization_step)
        quantized_direction = index_direction * self._quantization_step
        # TODO: raise an exception if duplicate directions found after quantization.

        return quantized_direction

    def quantize_float(self, direction: float) -> float:
        # direction between 0 and 360 degrees
        normalized_direction = cast(float, self.normalize(direction))
        index_direction = round(normalized_direction / self._quantization_step)
        quantized_direction = index_direction * self._quantization_step
        return quantized_direction

    @staticmethod
    def normalize(directions: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """ Normalize angle(s) expressed in degrees to the interval [-180°, 180°[

        :param directions: the real valued angle(s) expressed in degrees
        :returns: multiple(s) of quantization_step such that for each angle:
                  quantized_angle - quantization_step/2 < angle <=
                  quantized_angle + quantization_step/2
        """
        # angles between 0 and 360 degrees
        normalized_directions = directions % 360.
        # direction between -180 and +180 degrees
        if isinstance(normalized_directions, float):
            if normalized_directions >= 180.:
                normalized_directions -= 360.
        else:
            normalized_directions[normalized_directions >= 180.] -= 360.

        return normalized_directions
