# -*- coding: utf-8 -*-
""" Definition of the QuantizedDirectionsDict class and associated functions

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 13 Oct 2021

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
from typing import Any, List, Optional

import numpy as np

from .constrained_dict import ConstrainedDict
from .directions_quantizer import DirectionsQuantizer


class QuantizedDirectionsDict(ConstrainedDict):
    """ An abstract dictionary whose keys are directions expressed in degrees and quantized with
    some quantization step. The directions are expressed between -180 and +180 degrees.
    """

    def __init__(self, directions_quantization: Optional[float] = None) -> None:
        """ Constructor

        :param directions_quantization: the step to use for quantizing direction angles, for
                                        indexing purposes. Direction quantization is such that the
                                        0 degree direction is used as the origin, and any direction
                                        angle is transformed to the nearest quantized angle for
                                        indexing that direction in the radon transform.
        """
        super().__init__()
        self._sorted_directions: Optional[List[float]] = None
        self._quantizer = DirectionsQuantizer(directions_quantization)

    @property
    def quantizer(self) -> DirectionsQuantizer:
        """ :return: the quantizer to be applied to the directions defined in this dictionary """
        return self._quantizer

    @property
    def quantization_step(self) -> float:
        """ :return: the directions quantization step for this dictionary """
        return self._quantizer.quantization_step

    def constrained_key(self, key: float) -> float:
        # First check if key already accepted
        if key in self.keys():
            return key
        # Key not accepted yet. Check it.
        if not isinstance(key, float):
            raise TypeError('Keys for a QuantizedDirectionsDict must be float.')
        return self.quantizer.quantize_float(key)

    def __setitem__(self, key: float, value: Any) -> None:
        # _sorted_directions attribute must be reset in case a new item enters the dictionary
        self._sorted_directions = None
        ConstrainedDict.__setitem__(self, key, value)

    def __delitem__(self, key: float) -> None:
        # _sorted_directions attribute must be reset in case an item is deleted from the dictionary
        self._sorted_directions = None
        ConstrainedDict.__delitem__(self, key)

    @property
    def sorted_directions(self) -> List[float]:
        """ :returns: the list of the directions in the dictionary, sorted in ascending order.
        """
        if self._sorted_directions is None:
            self._sorted_directions = sorted(list(self.keys()))
        return self._sorted_directions

    @property
    def directions(self) -> np.ndarray:
        """ :return: the directions defined in this QuantizedDirectionsDict """
        return np.array(self.sorted_directions)

    @property
    def nb_directions(self) -> int:
        """ :return: the number of directions defined in this QuantizedDirectionsDict"""
        return len(self)
