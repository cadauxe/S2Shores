# -*- coding: utf-8 -*-
""" Definition of the CartoTile class and associated functions

:author: GIROS Alain
:created: 05/05/2021
"""
from typing import Tuple, List
import numpy as np  # @NoMove

from ..generic_utils.numpy_utils import split_samples

from .image_geometry_types import PointType


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
    nb_tiles_x = crop  # Possibly different from nb_tiles_y in the future
    nb_tiles_y = crop  # Possibly different from nb_tiles_x in the future

    x_samples_parts = split_samples(x_samples, nb_tiles_x)
    y_samples_parts = split_samples(y_samples, nb_tiles_y)
    for x_samples_part in x_samples_parts:
        for y_samples_part in y_samples_parts:
            subtile_def = (x_samples_part, y_samples_part)
            tiles_def.append(subtile_def)
    return tiles_def


class CartoTile:
    """ A tile is a subset of samples in a 2D space. Tiles are built by taking consecutive
    samples in the samples coordinates lists, which means that there is no constraint on the
    spatial distribution of these samples. It is up to the caller to impose these constraints
    by providing increasing or decreasing ordered lists of coordinates or whatever desired order,
    according to the needs.
    """

    def __init__(self, x_samples: np.ndarray, y_samples: np.ndarray) -> None:
        """ Define the samples belonging to the tile. These samples correspond to the cross
        product of the X and Y coordinates.

        :param x_samples: the X coordinates defining the tile samples
        :param y_samples: the Y coordinates defining the tile samples
        """

        self.x_samples = x_samples
        self.y_samples = y_samples

    @property
    def upper_left_sample(self) -> PointType:
        """ :returns: the coordinates of the upper left sample of the tile, assuming that
                      Y axis is decreasing from top to down.
        """
        return self.x_samples[0], self.y_samples[-1]

    @property
    def lower_right_sample(self) -> PointType:
        """ :returns: the coordinates of the loxer right sample of the tile, assuming that
                      Y axis is decreasing from top to down.
        """
        return self.x_samples[-1], self.y_samples[0]

    @property
    def nb_samples(self) -> int:
        """ :returns: the number of samples in the tile.
        """
        return len(self.x_samples) * len(self.y_samples)

    def __str__(self) -> str:
        msg = f' N: {self.nb_samples} = {len(self.x_samples)}*{len(self.y_samples)} '
        msg += f' X[{self.upper_left_sample[0]}, {self.lower_right_sample[0]}] *'
        msg += f' Y[{self.upper_left_sample[1]}, {self.lower_right_sample[1]}]'
        return msg
