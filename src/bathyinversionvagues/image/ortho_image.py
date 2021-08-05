# -*- coding: utf-8 -*-
""" Definition of the OrthoImage class

:author: GIROS Alain
:created: 17/05/2021
"""
from abc import ABC, abstractproperty, abstractmethod
from typing import Dict  # @NoMove

import numpy as np  # @NoMove

from .ortho_layout import OrthoLayout


class OrthoImage(ABC, OrthoLayout):
    """ An orthoimage is an image expressed in a cartographic system.
    """

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

    def build_infos(self) -> Dict[str, str]:
        """ :returns: a dictionary of metadata describing this ortho image
        """
        infos = {
            'sat': self.satellite,  #
            'AcquisitionTime': self.acquisition_time,  #
            'epsg': 'EPSG:' + str(self.epsg_code)}
        return infos

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
