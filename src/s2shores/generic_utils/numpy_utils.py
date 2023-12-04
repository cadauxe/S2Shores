# -*- coding: utf-8 -*-
""" Module gathering several functions using numpy for different data handling purposes

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 16/06/2021
"""
from functools import lru_cache
from hashlib import sha1
from typing import List

import numpy as np
import numpy.typing as npt


def sc_all(array: np.ndarray) -> bool:
    for value in array.flat:
        if not value:
            return False
    return True


def find(condition: np.ndarray) -> np.ndarray:
    res, = np.nonzero(np.ravel(condition))
    return res


def permute_axes(image: np.ndarray) -> np.ndarray:
    n1, n2, n3 = np.shape(image)
    permuted = np.zeros((n2, n3, n1))
    for i in np.arange(n1):
        permuted[:, :, i] = image[i, :, :]
    return permuted


@lru_cache()
def circular_mask(nb_lines: int, nb_columns: int, dtype: npt.DTypeLike) -> np.ndarray:
    """ Computes the inner disk centered on an image, to be used as a mask in some processing
    (radon transform for instance).

    Note: arguments are np.ndarray elements, but they are passed individually to allow for caching
    by lru_cache, which needs hashable arguments, and np.ndarray is not hashable.

    :param nb_lines: the number of lines of the image
    :param nb_columns: the number of columns of the image
    :param dtype: the numpy data type to use for creating the mask
    :returns: The inscribed disk as a 2D array with ones inside the centered disk and zeros outside
    """
    inscribed_diameter = min(nb_lines, nb_columns)
    radius = inscribed_diameter // 2
    circle_in_rect = np.zeros((nb_lines, nb_columns), dtype=dtype)
    center_line = nb_lines // 2
    center_column = nb_columns // 2
    for line in range(nb_lines):
        for column in range(nb_columns):
            dist_to_center = (line - center_line)**2 + (column - center_column)**2
            if dist_to_center <= radius**2:
                # used integral 1 to allow casting to the desired dtype
                circle_in_rect[line][column] = 1
    return circle_in_rect


def split_samples(samples: np.ndarray, nb_parts: int) -> List[np.ndarray]:
    """ Split a sequence or array in a number of almost equal sized parts

    :param samples: sequence or array to split in several parts
    :param nb_parts: number of parts to create from samples
    :returns: the list of created parts
    """
    parts = []
    part_length = int(len(samples) / nb_parts)
    for part_index in range(nb_parts):
        start_index = part_index * part_length
        stop_index = start_index + part_length
        if part_index == nb_parts - 1:
            stop_index = len(samples)
        parts.append(samples[start_index:stop_index])
    return parts


def dump_numpy_variable(variable: np.ndarray, variable_name: str) -> None:
    if variable is not None:
        print(f'{variable_name} {variable.shape} {variable.dtype}')
    print(variable)


class HashableNdArray:
    """ Hashable wrapper for ndarray objects.
    """

    def __init__(self, array: np.ndarray) -> None:
        """ Creates a new hashable object encapsulating an ndarray.

        :param array: The ndarray to encapsulate.
        """
        self._encapsulated_array = array
        self._hash = int(sha1(array.view()).hexdigest(), 16)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HashableNdArray):
            return False
        return all(self._encapsulated_array == other._encapsulated_array)

    def __hash__(self) -> int:
        return self._hash

    def unwrap(self) -> np.ndarray:
        """ Returns the encapsulated ndarray.
        """
        return self._encapsulated_array
