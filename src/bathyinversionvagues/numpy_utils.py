# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:45:01 2020

This is a bathymetry inversion package with all kinds of functions for
depth inversion. Initially designed for TODO

@author: erwinbergsma
         gregoirethoumyre
"""
from functools import lru_cache

import numpy as np


def sc_all(array: np.ndarray) -> bool:
    for x in array.flat:
        if not x:
            return False
    return True


def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res


def permute_axes(image: np.ndarray) -> np.ndarray:
    n1, n2, n3 = np.shape(image)
    permuted = np.zeros((n2, n3, n1))
    for i in np.arange(n1):
        permuted[:, :, i] = image[i, :, :]
    return permuted


@lru_cache()
def circular_mask(nb_lines: int, nb_columns: int, dtype: int) -> np.ndarray:
    """ Computes the inner disk centered on an image, to be used as a mask in some processing
    (radon transform for instance).

    Note: arguments are np.ndarray elements, but they are passed individually to allow for caching
    by lru_cache, which needs hashable arguments, and np.ndarray is not hashable.

    :param nb_lines: the number of lines of the image
    :param nb_columns: the number of columns of the image
    :returns: The inscribed disk as a 2D array with ones inside the centered disk and zeros outside
    """
    inscribed_diameter = min(nb_lines, nb_columns)
    radius = inscribed_diameter // 2
    image = np.zeros((nb_lines, nb_columns), dtype=dtype)
    center_line = nb_lines // 2
    center_column = nb_columns // 2
    for line in range(nb_lines):
        for column in range(nb_columns):
            dist_to_center = (line - center_line)**2 + (column - center_column)**2
            if dist_to_center <= radius**2:
                image[line][column] = 1  # used integral 1 to allow casting to the desired dtype
    return image
