# -*- coding: utf-8 -*-
""" Definition of the DirectionalArray class and associated functions

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 4 mars 2021

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
from typing import Any, Optional, Tuple, cast

import numpy as np

from .quantized_directions_dict import QuantizedDirectionsDict


# TODO: add a "symmetric" property, allowing to consider directions modulo pi as equivalent
# TODO: enable ordering such that circularity can be exposed (around +- 180° and 0°)
class DirectionalArray(QuantizedDirectionsDict):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._nb_samples = -1
        super().__init__(*args, **kwargs)

    def constrained_value(self, value: Any) -> Any:
        if not isinstance(value, np.ndarray) or value.ndim != 1:
            raise TypeError('Values for a DirectionalArray can only be 1D numpy arrays')
        if self._nb_samples < 0:
            self._nb_samples = value.size
        else:
            if value.size != self._nb_samples:
                msg = '1D arrays in a DirectionalArray must have the same size. Expected size'
                msg += f'(from first insert) is {self._nb_samples}, current is {value.size}'
                raise ValueError(msg)
        return value

    @property
    def nb_samples(self) -> int:
        """ :return: the length of each directional vector in this DirectionalArray"""
        return self._nb_samples

    def insert_from_arrays(self, array: np.ndarray, directions: np.ndarray) -> None:
        """ Insert a set of 1d arrays taken as columns of a 2D array, whose directions are provided
        in a 1d array of the same size.

        :param array: a 2D array containing directional vectors along each column
        :param directions: the set of directions in degrees associated to each array column.
        :raises TypeError: when array or directions have not the right number of dimensions
        :raises ValueError: when the number of dimensions is not consistent with the number of
                            columns in the array.
        """
        # Check that numpy arguments are of the right dimensions and consistent
        if array.ndim != 2:
            raise TypeError('array for a DirectionalArray must be a 2D numpy array')

        if directions.ndim != 1:
            raise TypeError('directions for a DirectionalArray must be a 1D numpy array')

        if directions.size != array.shape[1]:
            raise ValueError('directions size must be equal to the number of columns of the array')

        # FIXME: directions may be quantized.should we check them before insertion
        for index, direction in enumerate(directions.tolist()):
            self[direction] = array[:, index]
        if self.nb_directions != array.shape[1]:
            raise ValueError('dimensions after quantization has not the same number of elements '
                             f'({self.nb_directions}) than the number '
                             f'of columns in the array ({array.shape[1]})')

    def get_as_arrays(self,
                      directions: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns a 2D array with the requested directional values as columns

        :param directions: a vector of directions to store in the returned array, in their order
        :returns: a 2D array with the requested directional values as columns
        """
        if directions is None:
            selected_directions = np.array(self.sorted_directions)
        else:
            selected_directions_array = cast(np.ndarray, self.quantizer.quantize(directions))
            selected_directions = np.array(sorted(selected_directions_array.tolist()))

        # Build array by selecting the requested directions
        array_excerpt = np.empty((self.nb_samples, len(selected_directions)))
        for index, direction in enumerate(selected_directions):
            array_excerpt[:, index] = self[direction]
        return array_excerpt, selected_directions
