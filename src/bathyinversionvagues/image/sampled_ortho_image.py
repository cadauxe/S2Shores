# -*- coding: utf-8 -*-
""" Definition of the SampledOrthoImage class

:author: GIROS Alain
:created: 05/05/2021
"""
from typing import List, Optional  # @NoMove

from shapely.geometry import Polygon

import numpy as np

from ..image_processing.waves_image import WavesImage
from .carto_sampling import CartoSampling, build_tiling
from .image_geometry_types import PointType, MarginsType, ImageWindowType
from .ortho_stack import OrthoStack, FrameIdType


class SampledOrthoImage:
    """ This class makes the link between a CartoSampling and the image in which it is defined.
    """

    def __init__(self, ortho_stack: OrthoStack, carto_sampling: CartoSampling,
                 margins: MarginsType) -> None:
        """ Define the samples belonging to the subtile. These samples correspond to the cross
        product of the X and Y coordinates.

        :param ortho_stack: the orthorectified stack onto which the sampling is defined
        :param carto_sampling: the sampling of this SampledOrthoImage
        :param y_samples: the Y coordinates defining the samples of the subtile
        :param margins: the margins to consider around the samples to determine the image extent
        """
        self.ortho_stack = ortho_stack
        self.carto_sampling = carto_sampling
        self._margins = margins

        # col_start, line_start, nb_cols and nb_lines define the rectangle of pixels in image
        # coordinates which are just needed to process the subtile. No margins and no missing
        # lines or columns.
        self._line_start, _, self._col_start, _ = \
            self.ortho_stack.window_pixels(self.carto_sampling.upper_left_sample, self._margins)
        _, self._line_stop, _, self._col_stop = \
            self.ortho_stack.window_pixels(self.carto_sampling.lower_right_sample, self._margins)

    @classmethod
    def build_subtiles(cls, image: OrthoStack, nb_subtiles_max: int, step_x: float, step_y: float,
                       margins: MarginsType, roi: Optional[Polygon] = None) \
            -> List['SampledOrthoImage']:
        """ Class method building a set of SampledOrthoImage instances, forming a tiling of the
        specified orthorectifed image.

        :param image: the orthorectified image onto which the sampling is defined
        :param nb_subtiles_max: the meximum number of tiles to create
        :param step_x: the cartographic sampling to use along the X axis for building the tiles
        :param step_y: the cartographic sampling to use along the X axis for building the tiles
        :param margins: the margins to consider around the samples to determine the image extent
        :param roi: theroi for which bathymetry must be computed, if any.
        :returns: a list of SampledOrthoImage objects covering the orthorectfied image with the
                  specified sampling steps and margins.
        """
        ortho_sampling = image.get_samples_positions(step_x, step_y, margins, roi)
        subtiles_samplings = build_tiling(ortho_sampling, nb_subtiles_max)
        subtiles: List[SampledOrthoImage] = []
        for subtile_sampling in subtiles_samplings:
            subtiles.append(cls(image, subtile_sampling, margins))
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
