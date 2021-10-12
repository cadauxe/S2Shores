# -*- coding: utf-8 -*-
""" Definition of the DirectionalArray class and associated functions

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 4 mars 2021
"""
from typing import Optional  # @NoMove

import numpy as np

from .quantized_directions import (QuantizedDirections, linear_directions,
                                   DEFAULT_ANGLE_MAX, DEFAULT_ANGLE_MIN, DEFAULT_DIRECTIONS_STEP)


# TODO: add a "symmetric" property, allowing to consider directions modulo pi as equivalent
# TODO: enable ordering such that circularity can be exposed (around +- 180° and 0°)
class DirectionalArray:
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
        # Check that numpy arguments are of the right dimensions when provided
        if array.ndim != 2:
            raise TypeError('array for a DirectionalArray must be a 2D numpy array')
        self._array = array

        if directions.size != array.shape[1]:
            raise ValueError('directions size must be equal to the number of columns of the array')

        # TODO: implement the directions as the keys of a dictionary pointing to views in the array?
        self._directions = QuantizedDirections(directions, directions_step)

        if self.nb_directions != array.shape[1]:
            raise ValueError('dimensions after quantization has not the same number of elements '
                             f'({self.nb_directions}) than the number '
                             f'of columns in the array ({array.shape[1]})')

    @classmethod
    def create_empty(cls,
                     height: int,
                     directions: Optional[np.ndarray] = None,
                     directions_step: float = DEFAULT_DIRECTIONS_STEP) -> 'DirectionalArray':
        """ Creation of an empty DirectionalArray

        :param directions_step: the step to use for quantizing direction angles, for indexing
                                purposes. Direction quantization is such that the 0 degree direction
                                is used as the origin, and any direction angle is transformed to the
                                nearest quantized angle for indexing that direction in the radon
                                transform.
        :raises TypeError: when directions is not a 1D array
        """
        # Check that directions argument
        if directions is None:
            directions = linear_directions(DEFAULT_ANGLE_MIN, DEFAULT_ANGLE_MAX, directions_step)
        elif directions.ndim != 1:
            raise TypeError('dimensions for a DirectionalArray must be a 1D numpy array')

        array = np.empty((height, directions.size), dtype=np.float64)

        return cls(array, directions=directions, directions_step=directions_step)

    @property
    def directions(self) -> np.ndarray:
        """ :return: the directions defined in this DirectionalArray """
        return self._directions.values

    @property
    def quantization_step(self) -> float:
        """ :return: the directions defined in this DirectionalArray """
        return self._directions.quantizer._directions_step

    @property
    def nb_directions(self) -> int:
        """ :return: the number of directions defined in this DirectionalArray"""
        return self._directions.nb_directions

    @property
    def height(self) -> int:
        """ :return: the height of each directional vector in this DirectionalArray"""
        return self.get_as_array().shape[0]

    # TODO: allow float or array, return 1D or 2D array
    def values_for(self, direction: float) -> np.ndarray:
        direction_index = self._directions.find_index(direction)
        return self._array[:, direction_index]

    def values_at_index(self, direction_index: int) -> np.ndarray:
        return self.get_as_array()[:, direction_index]

    def set_at_direction(self, direction: float, array: np.ndarray) -> None:
        direction_index = self._directions.find_index(direction)
        self.get_as_array()[:, direction_index] = array

    def get_as_array(self, directions: Optional[np.ndarray] = None) -> np.ndarray:
        """ Returns a 2D array with the requested directional values as columns

        :param directions: a vector of directions to store in the returned array, in their order
        :returns: a 2D array with the requested directional values as columns
        """
        if directions is None:
            return self._array
        # TODO: use some method from QuantizedDirections
        quantized_directions, _ = self._directions.quantizer.quantize(directions)
        # Build array by selecting the requested directions
        array_excerpt = np.empty((self.height, quantized_directions.size))
        for i, direction in enumerate(quantized_directions):
            array_excerpt[:, i] = self.values_for(direction).reshape(self.height)
        return array_excerpt
