# -*- coding: utf-8 -*-
""" Definition of the DeltaTimeProvider abstract class and ConstantDeltaTimeProvider class

:author: GIROS Alain
:created: 02/08/2021
"""
from abc import abstractmethod
from typing import Optional


from ..image.image_geometry_types import PointType

from .localized_data_provider import LocalizedDataProvider


class DeltaTimeProvider(LocalizedDataProvider):
    """ A DeltaTimeProvider is a service able to provide the delta time at some position
    between two images. The points where delta time is requested are specified by their coordinates
    in the image SRS.
    """

    @abstractmethod
    def get_delta_time(self, point: PointType) -> float:
        """ Provides the delta time at some point expressed by its X and Y coordinates in some SRS.

        :param point: a tuple containing the X and Y coordinates in the SRS set for this provider
        :returns: the delta time between images at this point (m/s2).
        """


class ConstantDeltaTimeProvider(DeltaTimeProvider):
    """ A DeltaTimeProvider which provides a constant delta time for any position and images
    configurations.
    """

    def __init__(self, delta_time: float = 0.) -> None:
        super().__init__()
        self._constant_delta_time = delta_time

    def get_delta_time(self, _: Optional[PointType] = None) -> float:
        return self._constant_delta_time
