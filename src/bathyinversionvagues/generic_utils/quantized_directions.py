# -*- coding: utf-8 -*-
""" Definition of the DirectionalArray class and associated functions

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 12 Oct 2021
"""
import numpy as np

from .directions_quantizer import DirectionsQuantizer, DEFAULT_DIRECTIONS_STEP
DEFAULT_ANGLE_MIN = -180.
DEFAULT_ANGLE_MAX = 0.


def linear_directions(angle_min: float, angle_max: float, directions_step: float) -> np.ndarray:
    return np.linspace(angle_min, angle_max,
                       int((angle_max - angle_min) / directions_step),
                       endpoint=False)


class QuantizedDirections:
    def __init__(self, directions: np.ndarray,
                 directions_step: float = DEFAULT_DIRECTIONS_STEP) -> None:
        """ Constructor

        :param directions: a set of directions in degrees.
        :param directions_step: the step to use for quantizing direction angles, for indexing
                                purposes. Direction quantization is such that the 0 degree direction
                                is used as the origin, and any direction angle is transformed to the
                                nearest quantized angle for indexing purposes
        :raises TypeError: when directions is not a 1D array
        """
        if directions.ndim != 1:
            raise TypeError('dimensions for QuantizedDirections must be a 1D numpy array')

        self.quantizer = DirectionsQuantizer(directions_step)

        # TODO: implement the directions as the keys of a dictionary pointing to views in the array?
        self._directions = self._prepare_directions(directions)

    @property
    def values(self) -> np.ndarray:
        """ :return: the set of quantized directions"""
        return self._directions

    @property
    def nb_directions(self) -> int:
        """ :return: the number of directions"""
        return self._directions.size

    def _prepare_directions(self, directions: np.ndarray) -> np.ndarray:
        """ Quantize a set of directions and verify that no duplicates are created by quantization

        :param directions: an array of directions
        :returns: an array of quantized directions
        :raises ValueError: when the number of elements in the quantized array is different from
                            the number of elements in the input directions array
        """
        # TODO: consider defining direction_step from dimensions contents before quantizing
        # TODO: consider reordering the directions in increasing order
        quantized_directions = self.quantizer.quantize(directions)[0]
        unique_directions = np.unique(quantized_directions)
        if unique_directions.size != directions.size:
            raise ValueError('some dimensions values are too close to each other considering '
                             f'the dimensions quantization step: {self.quantizer._directions_step}°')
        return quantized_directions

    def find_index(self, direction: float) -> int:
        quantized_direction = self.quantizer.quantize(direction)
        direction_indexes = np.where(self._directions == quantized_direction[0])
        if direction_indexes[0].size != 1:
            raise ValueError(f'direction {direction} not found in the directional array')
        return direction_indexes[0]
