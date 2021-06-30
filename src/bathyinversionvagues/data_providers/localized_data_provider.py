# -*- coding: utf-8 -*-
""" Definition of the LocalizedDataProvider abstract class

:author: GIROS Alain
:created: 23/06/2021
"""
from abc import ABC
from typing import Optional, Tuple  # @NoMove

from osgeo import osr

from ..image.image_geometry_types import PointType


class LocalizedDataProvider(ABC):
    """ Base class for providers which deliver data depending on some location on Earth.
    It offers the ability to store the SRS which is used by the client of the provider for
    specifying a given point on Earth as well as the methods for transforming these coordinates
    into the working SRS of the provider.
    """

    def __init__(self) -> None:

        # Default provider SRS is set to EPSG:4326
        self.provider_srs = osr.SpatialReference()
        self.provider_srs.ImportFromEPSG(4326)

        # Default client SRS is set to EPSG:4326 as well
        self._client_epsg_code = 4326
        self.client_srs = osr.SpatialReference()
        self.client_srs.ImportFromEPSG(self._client_epsg_code)

        # Default SRS transformation does nothing
        self.client_to_provider_transform = osr.CoordinateTransformation(self.client_srs,
                                                                         self.provider_srs)

    @property
    def epsg_code(self) -> int:
        """ :returns: the epsg code of the SRS which will be used in subsequent client requests
        """
        return self._client_epsg_code

    @epsg_code.setter
    def epsg_code(self, value: int) -> None:
        self._client_epsg_code = value
        self.client_srs.ImportFromEPSG(value)
        self.client_to_provider_transform = osr.CoordinateTransformation(self.client_srs,
                                                                         self.provider_srs)

    def set_provider_epsg_code(self, value: int) -> None:
        """ Set the EPSG code of the SRS used by the provider to retrieve its own data

        :param value: EPSG code
        """
        self.provider_srs.ImportFromEPSG(value)
        self.client_to_provider_transform = osr.CoordinateTransformation(self.client_srs,
                                                                         self.provider_srs)

    def transform_point(self, point: PointType, altitude: float) -> Tuple[float, float, float]:
        """ Transform a point in 3D from the client SRS to the provider SRS

        :param point: (X, Y) coordinates of the point in the client SRS
        :param altitude: altitude of the point in the client SRS
        :returns: 3D coordinates in the provider SRS corresponding to the point. Meaning of
                  these coordinates depends on the provider SRS: (longitude, latitude, height) for
                  geographical SRS or (X, Y, height) for cartographic SRS.

        """
        return self.client_to_provider_transform.TransformPoint(*point, altitude)
