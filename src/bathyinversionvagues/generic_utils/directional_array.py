# -*- coding: utf-8 -*-
""" Definition of the DirectionalArray class and associated functions

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 4 mars 2021
"""
from typing import Optional, Any

import numpy as np

from .directions_quantizer import DEFAULT_DIRECTIONS_STEP
from .directions_quantizer import DirectionsQuantizer
from .quantized_directions_dict import QuantizedDirectionsDict


# TODO: add a "symmetric" property, allowing to consider directions modulo pi as equivalent
# TODO: enable ordering such that circularity can be exposed (around +- 180° and 0°)
class DirectionalArray(QuantizedDirectionsDict):
    def __init__(self, array: np.ndarray, directions: np.ndarray,
                 directions_step: float = DEFAULT_DIRECTIONS_STEP) -> None:
        """ Constructor

        :param array: a 2D array containing directional vectors along each column
        :param directions: the set of directions in degrees associated to each array column.
        :param directions_step: the step to use for quantizing direction angles, for indexing
                                purposes. Direction quantization is such that the 0 degree direction
                                is used as the origin, and any direction angle is transformed to the
                                nearest quantized angle for indexing that direction in the radon
                                transform.
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

        self._array_length: Optional[int] = None
        self._quantizer = DirectionsQuantizer(directions_step)
        super().__init__()
        for index, direction in enumerate(directions.tolist()):
            self[direction] = array[:, index]
        if self.nb_directions != array.shape[1]:
            raise ValueError('dimensions after quantization has not the same number of elements '
                             f'({self.nb_directions}) than the number '
                             f'of columns in the array ({array.shape[1]})')

    def constrained_value(self, value: Any) -> Any:
        if not isinstance(value, np.ndarray) or value.ndim != 1:
            raise TypeError('Values for a DirectionalArray can only be 1D numpy arrays')
        if self._array_length is None:
            self._array_length = value.size
        else:
            if value.size != self._array_length:
                msg = '1D arrays in a DirectionalArray must have the same size. Expected size'
                msg += f'(from first insert) is {self._array_length}, current is {value.size}'
                raise ValueError(msg)
        return value

    @property
    def quantizer(self) -> DirectionsQuantizer:
        return self._quantizer

    @quantizer.setter
    def quantizer(self, quantizer: DirectionsQuantizer) -> None:
        self._quantizer = quantizer

    @property
    def height(self) -> int:
        """ :return: the height of each directional vector in this DirectionalArray"""
        return self._array_length

    def get_as_array(self, directions: Optional[np.ndarray] = None) -> np.ndarray:
        """ Returns a 2D array with the requested directional values as columns

        :param directions: a vector of directions to store in the returned array, in their order
        :returns: a 2D array with the requested directional values as columns
        """
        if directions is None:
            selected_directions = self.sorted_directions
        else:
            selected_directions_array = self.quantizer.quantize(directions)
            selected_directions = sorted(selected_directions_array.tolist())

        # Build array by selecting the requested directions
        array_excerpt = np.empty((self._array_length, len(selected_directions)))
        for i, direction in enumerate(selected_directions):
            array_excerpt[:, i] = self[direction].reshape(self._array_length)
        return array_excerpt
