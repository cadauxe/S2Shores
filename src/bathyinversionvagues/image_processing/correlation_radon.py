# -*- coding: utf-8 -*-
"""
module -- Class encapsulating operations on the radon transform of a correlation matrix
"""

from typing import Optional

import numpy as np

from ..waves_exceptions import NoRadonTransformError

from .correlation_image import CorrelationImage
from .waves_radon import WavesRadon
from .waves_sinogram import WavesSinogram


class CorrelationRadon(WavesRadon):
    """
    This class extends class WavesRadon and is used for radon transform computed from a centered
    correlation matrix
    It is very important that circle=True has been chosen to compute radon matrix since we read
    values in meters from the axis of the sinogram
    """

    def __init__(self, image: CorrelationImage, directions_step: float = 1.,
                 weighted: bool = False) -> None:
        super().__init__(image, directions_step, weighted)

    def get_sinogram_maximum_variance(self,
                                      directions: Optional[np.ndarray] = None) -> (
            WavesSinogram, float):
        directions = self.directions if directions is None else directions
        maximum_variance = None
        sinogram_maximum_variance = None
        index_maximum_variance_direction = None
        for result_index, sinogram_index in enumerate(directions):
            sinogram = self.sinograms[sinogram_index]
            if maximum_variance is None:
                maximum_variance = sinogram.variance
                sinogram_maximum_variance = sinogram
                index_maximum_variance_direction = result_index
            elif maximum_variance < sinogram.variance:
                maximum_variance = sinogram.variance
                sinogram_maximum_variance = sinogram
                index_maximum_variance_direction = result_index
        return (sinogram_maximum_variance, directions[index_maximum_variance_direction])
