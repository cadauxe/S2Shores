# -*- coding: utf-8 -*-
""" Definition of the DistToShoreProvider abstract class

:author: GIROS Alain
:created: 23/06/2021
"""
from abc import abstractmethod
from pathlib import Path

import xarray as xr  # @NoMove
import numpy as np

from ..image.image_geometry_types import PointType

from .localized_data_provider import LocalizedDataProvider


class DisToShoreProvider(LocalizedDataProvider):
    """ A distoshore provider is a service which is able to provide the distance to shore of a
    point specified by its coordinates in some SRS.
    """

    @abstractmethod
    def get_distance(self, point: PointType) -> float:
        """ Provides the distance to shore of a point in kilometers.

        :param point: a tuple containing the X and Y coordinates in the SRS set for this provider
        :returns: the distance to shore in kilometers (positive over water, zero on ground,
                  positive infinity if unknown).
        """


class InfinityDisToShoreProvider(DisToShoreProvider):
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


class NetCDFDisToShoreProvider(DisToShoreProvider):
    """ A DistToShoreProvider which provides the distance to shore when it is stored in a
    'disToShore' layer of a netCDF file.
    """

    # FIXME: EPSG code needed because no SRS retrieved from the NetCDF file at this time.
    def __init__(self, distoshore_file_path: Path, distoshore_epsg_code: int) -> None:
        """ Create a NetCDFDisToShoreProvider object and set necessary informations

        :param distoshore_file_path: full path of a netCDF file containing the distance to shore
                                     to be used by this provider.
        :param distoshore_epsg_code: the EPSG code of the SRS used in the NetCDF file.
        """
        super().__init__()
        self.set_provider_epsg_code(distoshore_epsg_code)
        self._distoshore_xarray = xr.open_dataset(distoshore_file_path)

    def get_distance(self, point: PointType) -> float:
        """ Provides the distance to shore of a point in kilometers.

        :param point: a tuple containing the X and Y coordinates in the SRS of the client
        :returns: the distance to the nearest shore (km)
        """
        provider_point = self.transform_point(point, 0.)
        distance_xr_dataset = self._distoshore_xarray.sel(y=provider_point[1],
                                                          x=provider_point[0], method='nearest')
        return float(distance_xr_dataset['disToShore'].values)
