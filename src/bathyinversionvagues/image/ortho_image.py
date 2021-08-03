# -*- coding: utf-8 -*-
""" Definition of the OrthoImage class

:author: GIROS Alain
:created: 17/05/2021
"""
from abc import ABC, abstractproperty, abstractmethod
from typing import Tuple, Dict  # @NoMove

import numpy as np  # @NoMove

from ..generic_utils.tiling_utils import modular_sampling

from .geo_transform import GeoTransform
from .image_geometry_types import MarginsType, PointType, ImageWindowType, GdalGeoTransformType


class OrthoImage(ABC):
    """ An orthoimage is an image expressed in a cartographic system. This class is an abstract
    class linking the cartographic and the image extents of the image.
    """

    def __init__(self, nb_columns: int, nb_lines: int, projection: str,
                 gdal_geotransform: GdalGeoTransformType) -> None:
        """ Constructor.

        :param nb_columns: the number of columns of this image
        :param nb_lines: the number of lines of this image
        :param projection: the projection of this orthorectified image, as a wkt.
        :param gdal_geotransform: the GDAL geotransform allowing to transform cartographic
                                  coordinates into image coordinates and reciprocally.
        """
        # Extract the relevant information from one of the jp2 images
        self._nb_columns = nb_columns
        self._nb_lines = nb_lines
        self._projection = projection
        self._geo_transform = GeoTransform(gdal_geotransform)

        # Get georeferenced extent of the whole image
        self.upper_left_x, self.upper_left_y = self._geo_transform.projected_coordinates(0., 0.)
        self.lower_right_x, self.lower_right_y = self._geo_transform.projected_coordinates(
            self._nb_columns, self._nb_lines)

    @property
    @abstractproperty
    def short_name(self) -> str:
        """ :returns: the short image name
        """

    @property
    @abstractproperty
    def satellite(self) -> str:
        """ :returns: the satellite identifier
        """

    @property
    @abstractproperty
    def acquisition_time(self) -> str:
        """ :returns: the acquisition time of the image
        """

    @property
    def epsg_code(self) -> int:
        """ :returns: the epsg code of the projection
        """
        return int(self._projection.split(',')[-1][1:-3])

    def build_infos(self) -> Dict[str, str]:
        """ :returns: a dictionary of metadata describing this ortho image
        """
        infos = {
            'sat': self.satellite,  #
            'AcquisitionTime': self.acquisition_time,  #
            'epsg': 'EPSG:' + str(self.epsg_code)}
        return infos

    # TODO: define steps default values based on resolution
    def get_samples_positions(self, step_x: float, step_y: float, local_margins: MarginsType
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """ x_samples, y_samples are the coordinates  of the final samples in georeferenced system
        sampled from a starting position with steps 'DXP' and 'DYP'

        :param step_x: the cartographic sampling to use along the X axis to sample this image
        :param step_y: the cartographic sampling to use along the X axis to sample this image
        :param local_margins: the margins to consider around the samples
        :returns: the chosen samples specified by the cross product of X samples and Y samples
        """
        # Compute all the sampling X and Y coordinates falling inside the image domain
        left_sample_index, right_sample_index = modular_sampling(self.upper_left_x,
                                                                 self.lower_right_x,
                                                                 step_x)

        bottom_sample_index, top_sample_index = modular_sampling(self.lower_right_y,
                                                                 self.upper_left_y,
                                                                 step_y)
        x_samples = np.arange(left_sample_index, right_sample_index + 1) * step_x
        y_samples = np.arange(bottom_sample_index, top_sample_index + 1) * step_y

        # Adding half resolution to point at the pixel center
        x_samples += self._geo_transform.x_resolution / 2.
        y_samples += self._geo_transform.y_resolution / 2.

        acceptable_samples_x = []
        for x_coord in x_samples:
            for y_coord in y_samples:
                line_start, line_stop, col_start, col_stop = self.window_pixels((x_coord, y_coord),
                                                                                local_margins)
                if (line_start >= 0 and line_stop < self._nb_lines and
                        col_start >= 0 and col_stop < self._nb_columns):
                    acceptable_samples_x.append(x_coord)
                    break

        acceptable_samples_y = []
        for y_coord in y_samples:
            for x_coord in acceptable_samples_x:
                line_start, line_stop, col_start, col_stop = self.window_pixels((x_coord, y_coord),
                                                                                local_margins)
                if (line_start >= 0 and line_stop < self._nb_lines and
                        col_start >= 0 and col_stop < self._nb_columns):
                    acceptable_samples_y.append(y_coord)
                    break

        x_samples = np.array(acceptable_samples_x)
        y_samples = np.array(acceptable_samples_y)

        return x_samples, y_samples

    def window_pixels(self, point: PointType, margins: MarginsType,
                      line_start: int = 0, col_start: int = 0) -> ImageWindowType:
        """ Given a point defined in the projected domain, computes a rectangle of pixels centered
        on the pixel containing this point and taking into account the specified margins.
        No check is done at this level to verify that the rectangle is contained within the pixels
        space.

        :param point: the X and Y coordinates of the point
        :param margins: the margins to consider around the point in order to build the window.
        :param line_start: line number in the image from which the window coordinates are computed
        :param col_start: column number in the image from which the window coordinates are computed
        :returns: the window as a tuple of four coordinates relative to line_start and col_start:
                  - start and stop lines (both included) in the image space defining the window
                  - start and stop columns  (both included) in the image space defining the window
        """
        # define the sub window domain in utm
        window_proj = [point[0] - margins[0], point[0] + margins[1],
                       point[1] - margins[2], point[1] + margins[3]]

        # compute the sub window domain in pixels
        window_col_start, window_line_start = self._geo_transform.image_coordinates(window_proj[0],
                                                                                    window_proj[3])
        window_col_stop, window_line_stop = self._geo_transform.image_coordinates(window_proj[1],
                                                                                  window_proj[2])
        window_pix = (int(window_line_start) - line_start,
                      int(window_line_stop) - line_start,
                      int(window_col_start) - col_start,
                      int(window_col_stop) - col_start)
        return window_pix

    @abstractmethod
    def read_pixels(self, band_id: str, line_start: int, line_stop: int,
                    col_start: int, col_stop: int) -> np.ndarray:
        """ Read a rectangle of pixels from a specific band of this image.

        :param band_id: the identifier of the spectral band
        :param line_start: the image line where the rectangle begins
        :param line_stop: the image line where the rectangle stops
        :param col_start: the image column where the rectangle begins
        :param col_stop: the image column where the rectangle stops
        :returns: the rectangle of pixels as an array
        """
