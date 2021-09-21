# -*- coding: utf-8 -*-
""" Definition of the OrthoImage class

:author: GIROS Alain
:created: 17/05/2021
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict  # @NoMove

from osgeo import gdal

from ..image_processing.waves_image import WavesImage

from .ortho_layout import OrthoLayout


class OrthoImage(ABC, OrthoLayout):
    """ An orthoimage is an image expressed in a cartographic system.
    """

    @property
    @abstractmethod
    def short_name(self) -> str:
        """ :returns: the short image name
        """

    @property
    @abstractmethod
    def satellite(self) -> str:
        """ :returns: the satellite identifier
        """

    @property
    @abstractmethod
    def acquisition_time(self) -> str:
        """ :returns: the acquisition time of the image
        """

    @property
    def spatial_resolution(self) -> float:
        """ :returns: the spatial resolution of the image (m)
        """
        return self._geo_transform.resolution

    def build_infos(self) -> Dict[str, str]:
        """ :returns: a dictionary of metadata describing this ortho image
        """
        infos = {
            'sat': self.satellite,  #
            'AcquisitionTime': self.acquisition_time,  #
            'epsg': 'EPSG:' + str(self.epsg_code)}
        return infos

    @abstractmethod
    def get_image_file_path(self, band_id: str) -> Path:
        """ Provides the full path to the file containing a given band of this orthoimage

        :param band_id: the identifier of the spectral band (e.g. 'B02')
        :returns: the path to the file containing the spectral band
        """

    @abstractmethod
    def get_band_index_in_file(self, band_id: str) -> int:
        """ Provides the index in the image file of a given band of this orthoimage

        :param band_id: the identifier of the spectral band (e.g. 'B02')
        :returns: the index of the band in the file where it is contained
        """

    def read_pixels(self, band_id: str, line_start: int, line_stop: int,
                    col_start: int, col_stop: int) -> WavesImage:
        """ Read a rectangle of pixels from a specific band of this image.

        :param band_id: the identifier of the  band to read
        :param line_start: the image line where the rectangle begins
        :param line_stop: the image line where the rectangle stops
        :param col_start: the image column where the rectangle begins
        :param col_stop: the image column where the rectangle stops
        :returns: a sub image
        """
        image_dataset = gdal.Open(str(self.get_image_file_path(band_id)))
        image = image_dataset.GetRasterBand(self.get_band_index_in_file(band_id))
        nb_cols = col_stop - col_start + 1
        nb_lines = line_stop - line_start + 1
        pixels = image.ReadAsArray(col_start, line_start, nb_cols, nb_lines)
        # release dataset
        image_dataset = None
        return WavesImage(pixels, self.spatial_resolution)
