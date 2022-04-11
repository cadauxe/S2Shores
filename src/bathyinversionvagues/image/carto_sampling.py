# -*- coding: utf-8 -*-
""" Definition of the CartoSampling class and associated functions

:author: GIROS Alain
:created: 05/05/2021
"""
from typing import Tuple, List
import numpy as np  # @NoMove

from ..generic_utils.numpy_utils import split_samples
from ..waves_exceptions import WavesIndexingError

from .image_geometry_types import PointType


# TODO: define this as a class method of CartoSampling
def build_tiling(x_samples: np.ndarray, y_samples: np.ndarray,
                 nb_tiles_max: int = 1) -> List[Tuple[np.ndarray, np.ndarray]]:
    """ Build a tiling of the provided samples. The number of created tiles may be lower
    than the requested number of tiles, if this number is not a square number.

    :param x_samples: the samples X coordinates, no order imposed.
    :param y_samples: the samples Y coordinates, no order imposed.
    :param nb_tiles_max: the maximum number of tiles in the tiling

    :returns: a list of samples defining a set of tiles covering the provided samples.
    """

    tiles_def = []
    # Full samples cropped in crop*crop tiles
    crop = int(np.sqrt(nb_tiles_max))
    nb_tiles_x = min(crop, x_samples.size)  # Possibly different from nb_tiles_y in the future
    nb_tiles_y = min(crop, y_samples.size)  # Possibly different from nb_tiles_x in the future
    x_samples_parts = split_samples(x_samples, nb_tiles_x)
    y_samples_parts = split_samples(y_samples, nb_tiles_y)
    for x_samples_part in x_samples_parts:
        for y_samples_part in y_samples_parts:
            subtile_def = (x_samples_part, y_samples_part)
            tiles_def.append(subtile_def)
    return tiles_def


# TODO: create an iterator over sampled points and use it
class CartoSampling:
    """ A carto sampling is a subset of samples in a 2D space. It is built by taking consecutive
    samples in some samples coordinates lists, which means that there is no constraint on the
    spatial distribution of these samples. It is up to the caller to impose these constraints
    by providing increasing or decreasing ordered lists of coordinates or whatever desired order,
    according to the needs.
    """

    def __init__(self, x_samples: np.ndarray, y_samples: np.ndarray) -> None:
        """ Define the samples defining this carto sampling. These samples correspond to the cross
        product of the X and Y coordinates.

        :param x_samples: the X coordinates defining the carto sampling
        :param y_samples: the Y coordinates defining the carto sampling
        """

        self._x_samples = x_samples
        self._y_samples = y_samples

    @property
    def upper_left_sample(self) -> PointType:
        """ :returns: the coordinates of the upper left sample of this sampling, assuming that
                      Y axis is decreasing from top to down.
        """
        return self._x_samples[0], self._y_samples[-1]

    @property
    def lower_right_sample(self) -> PointType:
        """ :returns: the coordinates of the loxer right sample of this sampling, assuming that
                      Y axis is decreasing from top to down.
        """
        return self._x_samples[-1], self._y_samples[0]

    @property
    def nb_samples(self) -> int:
        """ :returns: the number of samples in this carto sampling.
        """
        return len(self._x_samples) * len(self._y_samples)

    @property
    def shape(self) -> Tuple[int, int]:
        """ :returns: the shape of the 2D sampling.
        """
        return self._y_samples.shape[0], self._x_samples.shape[0]

    def index_point(self, point: PointType) -> Tuple[int, int]:
        """ Retrieve the indexes of the coordinates of a point in the X and Y samples

        :param point: a point in 2D, whose coordinates must be retrieved in the sampling
        :returns: the indexes of X and Y in the sampling definitions
        :raises WavesIndexingError: when one coordinate of the point is undefined in the sampling
        """
        x_index = np.where(self._x_samples == point[0])
        if x_index[0].size == 0:
            msg_err = f'X coordinate: { point[0]} undefined in x_samples: {self._x_samples}'
            raise WavesIndexingError(msg_err)
        y_index = np.where(self._y_samples == point[1])
        if y_index[0].size == 0:
            msg_err = f'Y coordinate: { point[1]} undefined in y_samples: {self._y_samples}'
            raise WavesIndexingError(msg_err)
        return x_index[0][0], y_index[0][0]

    def __str__(self) -> str:
        msg = f' N: {self.nb_samples} = {len(self._y_samples)}*{len(self._x_samples)} '
        msg += f' X[{self.upper_left_sample[0]}, {self.lower_right_sample[0]}] *'
        msg += f' Y[{self.upper_left_sample[1]}, {self.lower_right_sample[1]}]'
        return msg
