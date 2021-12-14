# -*- coding: utf-8 -*-
""" Definition of the RoiProvider classes

:author: GIROS Alain
:created: 07/12/2021
"""
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Optional  # @NoMove

from osgeo import ogr
from shapely.geometry import Polygon, Point, MultiPolygon

from ..image.image_geometry_types import PointType
from .localized_data_provider import LocalizedDataProvider


class RoiProvider(ABC, LocalizedDataProvider):
    """ A Roi provider is a service which is able to test if a point specified by its coordinates
    in some SRS in inside a Region Of Interest expressed as a set of polygons defined in another
    SRS.
    """

    @abstractmethod
    def contains(self, point: PointType) -> bool:
        """ Test if a point is inside the ROI

        :param point: a tuple containing the X and Y coordinates in the SRS set for this provider
        :returns: True if the point lies inside the ROI
        """


class VectorFileRoiProvider(RoiProvider):
    """ A RoiProvider where the ROI is defined by a vector file in some standard format.
    """

    def __init__(self, vector_file_path: Path) -> None:
        """ Create a NetCDFDisToShoreProvider object and set necessary informations

        :param vector_file_path: full path of a vector file containing the ROI as a non empty set of
                                 polygons
        """
        super().__init__()

        self._polygons: Optional[MultiPolygon] = None
        self._vector_file_path = vector_file_path

    def contains(self, point: PointType) -> bool:
        if self._polygons is None:
            self._load_polygons()
        tranformed_point = Point(*self.transform_point(point, 0.))
        return self._polygons.contains(tranformed_point)

    def _load_polygons(self) -> None:
        """ Read the vector file and loads the polygons contained in its first layer
        """
        polygons = []
        dataset = ogr.Open(str(self._vector_file_path))
        layer = dataset.GetLayerByIndex(0)
        self.provider_epsg_code = int(layer.GetSpatialRef().GetAuthorityCode(None))
        for i in range(layer.GetFeatureCount()):
            feature = layer.GetFeature(i)
            polygon_ring = feature.GetGeometryRef().GetGeometryRef(0)

            # We use shapely Polygon in order to circumvent a core dump when using OGR with dask
            polygon = Polygon(polygon_ring.GetPoints())
            polygons.append(polygon)
        self._polygons = MultiPolygon(polygons)
