# -*- coding: utf-8 -*-
""" Definition of the DistToShoreProvider abstract class

:author: GIROS Alain
:created: 23/06/2021
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..image.image_geometry_types import PointType


class DisToShoreProvider(ABC):
    """ A distoshore provider is a service which is able to provide the distance to shore of a
    point specified by its coordinates in some SRS.
    """

    def __init__(self) -> None:
        self._epsg_code: Optional[int] = None

    @property
    def epsg_code(self) -> Optional[int]:
        """ :returns: the epsg code of the SRS currently set for this provider
        """
        return self._epsg_code

    @epsg_code.setter
    def epsg_code(self, value: int) -> None:
        self._epsg_code = value

    @abstractmethod
    def get_distance(self, point: PointType) -> float:
        """ Provides the distance to shore of a point in kilometers.

        :param point: a tuple containing the X and Y coordinates in the SRS set for this provider
        :returns: the distance to shore in kilometers (positive over water, zero on ground,
                  positive infinity if unknown).
        """


class DefaultDisToShoreProvider(DisToShoreProvider):
    """ A DistToShoreProvider which provides infinity distance to any request, ensuring that any
    point is always considered on water.
    """

    def get_distance(self, point: PointType) -> float:
        """ Provides the distance to shore of a point in kilometers.

        :param point: a tuple containing the X and Y coordinates in the SRS set for this provider
        :returns: positive infinity for any position
        """
        _ = point
        return np.Infinity
