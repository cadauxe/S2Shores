# -*- coding: utf-8 -*-
""" Definition of the SampledOrthoImage class

:author: GIROS Alain
:created: 05/05/2021
"""
from typing import List, Optional  # @NoMove

from shapely.geometry import Polygon

import numpy as np

from ..image_processing.waves_image import WavesImage
from .carto_tile import CartoTile, build_tiling
from .image_geometry_types import PointType, MarginsType, ImageWindowType
from .ortho_stack import OrthoStack, FrameIdType


class SampledOrthoImage(CartoTile):
    """ This class makes the link between a CartoTile and the image in which it is defined.
    """

    def __init__(self, ortho_stack: OrthoStack, x_samples: np.ndarray, y_samples: np.ndarray,
                 margins: MarginsType) -> None:
        """ Define the samples belonging to the subtile. These samples correspond to the cross
        product of the X and Y coordinates.

        :param ortho_stack: the orthorectified stack onto which the sampling is defined
        :param x_samples: the X coordinates defining the samples of the subtile
        :param y_samples: the Y coordinates defining the samples of the subtile
        :param margins: the margins to consider around the samples to determine the image extent
        """
        super().__init__(x_samples, y_samples)
        self.ortho_stack = ortho_stack
        self._margins = margins

        # col_start, line_start, nb_cols and nb_lines define the rectangle of pixels in image
        # coordinates which are just needed to process the subtile. No margins and no missing
        # lines or columns.
        self._line_start, _, self._col_start, _ = \
            self.ortho_stack.window_pixels(self.upper_left_sample, self._margins)
        _, self._line_stop, _, self._col_stop = \
            self.ortho_stack.window_pixels(self.lower_right_sample, self._margins)

    @classmethod
    def build_subtiles(cls, image: OrthoStack, nb_subtiles_max: int, step_x: float, step_y: float,
                       margins: MarginsType, roi_limit: Optional[Polygon] = None) \
            -> List['SampledOrthoImage']:
        """ Class method building a set of SampledOrthoImage instances, forming a tiling of the
        specified orthorectifed image.

        :param image: the orthorectified image onto which the sampling is defined
        :param nb_subtiles_max: the meximum number of tiles to create
        :param step_x: the cartographic sampling to use along the X axis for building the tiles
        :param step_y: the cartographic sampling to use along the X axis for building the tiles
        :param margins: the margins to consider around the samples to determine the image extent
        :returns: a list of SampledOrthoImage objects covering the orthorectfied image with the
                  specified sampling steps and margins.
        """
        x_samples, y_samples = image.get_samples_positions(step_x, step_y, margins, roi_limit)

        subtiles_def = build_tiling(x_samples, y_samples, nb_subtiles_max)
        subtiles: List[SampledOrthoImage] = []
        for subtile_def in subtiles_def:
            subtiles.append(cls(image, *subtile_def, margins))
        return subtiles

    def read_pixels(self, frame_id: FrameIdType) -> WavesImage:
        """ Read the whole rectangle of pixels corresponding to this SampledOrthoImage
        retrieved from a specific frame of the orthorectified stack.

        :param frame_id: the identifier of the frame in the stack
        :returns: the rectangle of pixels as an array
        """
        return self.ortho_stack.read_pixels(frame_id,
                                            self._line_start, self._line_stop,
                                            self._col_start, self._col_stop)

    def window_extent(self, carto_point: PointType) -> ImageWindowType:
        """ Given a point defined in the projected domain, computes a rectangle of pixels centered
        on the pixel containing this point and taking into account the SampledOrthoImage margins.

        :param carto_point: the X and Y coordinates of the point
        :returns: the window as a tuple of four coordinates relative to line_start and col_start of
                  this SampledOrthoImage
        """
        return self.ortho_stack.window_pixels(carto_point, self._margins,
                                              self._line_start, self._col_start)

    def __str__(self) -> str:
        msg = super().__str__()
        msg += f' C[{self._col_start}, {self._col_stop}] * L[{self._line_start}, {self._line_stop}]'
        return msg
