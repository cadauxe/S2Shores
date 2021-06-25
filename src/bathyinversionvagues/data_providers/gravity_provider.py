# -*- coding: utf-8 -*-
""" Definition of the GravityProvider abstract class and ConstantGravityProvider class

:author: GIROS Alain
:created: 25/06/2021
"""
from abc import ABC, abstractmethod
import math

from typing import Optional

from ..image.image_geometry_types import PointType


class GravityProvider(ABC):
    """ A GravityProvider provider is a service able to provide the gravity at different places
    and altitudes on earth. The points where gravity is requested are specified by coordinates
    in some SRS.
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
    def get_gravity(self, point: PointType, altitude: float) -> float:
        """ Provides the gravity at some point expressed by its X, Y and H coordinates in some SRS.

        :param point: a tuple containing the X and Y coordinates in the SRS set for this provider
        :param altitude: the altitude of the point in the SRS set for this provider
        :returns: the acceleration due to gravity at this point (m/s2).
        """


class ConstantGravityProvider(GravityProvider):
    """ A GravityProvider which provides the mean accelation of the gravity on Earth.
    """

    def get_gravity(self, point: PointType, altitude: float) -> float:
        _ = point
        _ = altitude
        # TODO: replace return value by 9.80665
        return 9.81


class LatitudeVaryingGravityProvider(GravityProvider):
    """ A GravityProvider which provides the acceleration of the gravity depending on the
    latitude of the point on Earth.
    """

    g_poles = 9.832
    g_45 = 9.806
    g_equator = 9.780
    g_mean = (g_poles - g_equator) / 2.

    def get_gravity(self, point: PointType, altitude: float) -> float:
        # FIXME: define SRS transform in base class
        latitude, _, _ = self.transform_point(point, altitude, target_epsg=4326)
        gravity = self.g_45 - self.g_mean * math.cos(latitude * math.pi / 90)
        return gravity
