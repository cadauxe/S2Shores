# -*- coding: utf-8 -*-
"""
module -- Class encapsulating an image onto which waves estimation will be made


:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 7 mars 2021
"""
from typing import Optional, Tuple

import numpy as np

from ..generic_utils.numpy_utils import circular_mask

from .shoresutils import funDetrend_2d, funSmooth2


# TODO: add the management of the image position
# TODO: add the azimuth of the image, if known
# TODO: add the possibility to apply several preprocessing filters
class WavesImage():
    def __init__(self, pixels: np.ndarray, resolution: float,
                 detrend: bool = True, smoothing: Optional[Tuple[int, int]] = None) -> None:
        """ Constructor

        :param pixels: a 2D array containing an image over water
        :param resolution: Image resolution in meters
        :param detrend: True (default) if an optional detrend must be applied on the image.
        """
        self.resolution = resolution
        self.pixels = pixels

        # pre-processing at creation time
        self._preprocess(detrend, smoothing)

    def _preprocess(self, detrend: bool = True,
                    smoothing: Optional[Tuple[int, int]] = None) -> None:
        # Detrending
        if detrend:
            self.detrend()

        # Background filtering
        if smoothing is not None:
            # FIXME: pixels necessary for smoothing are not taken into account, thus
            # zeros are introduced at the borders of the window.
            smoothed_pixels = funSmooth2(self.pixels, smoothing[0], smoothing[1])
            self.pixels = self.pixels - smoothed_pixels
            # Remove tendency possibly introduced by smoothing, specially on the shore line
            self.pixels = funDetrend_2d(self.pixels)

            # Disk masking
            # self.pixels = self.pixels * self.circle_image

    @property
    def sampling_frequency(self) -> float:
        """ :returns: The spatial sampling frequency of this image (m-1)"""
        return 1. / self.resolution

    @property
    def energy(self) -> float:
        """ :returns: The energy of the image"""
        return np.sum(self.pixels * self.pixels)

    @property
    def energy_inner_disk(self) -> np.ndarray:
        """ :returns: The energy of the image within its inscribed disk"""

        return np.sum(self.pixels * self.pixels * self.circle_image)

    @property
    def circle_image(self) -> np.ndarray:
        """ :returns: The inscribed disk"""
        # FIXME: Ratio of the disk area on the chip area should be closer to PI/4 (0.02 difference)
        return circular_mask(self.pixels.shape[0], self.pixels.shape[1], self.pixels.dtype)

    def detrend(self):
        self.pixels = funDetrend_2d(self.pixels)
