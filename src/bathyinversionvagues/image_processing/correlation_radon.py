# -*- coding: utf-8 -*-
"""
module -- Class encapsulating operations on the radon transform of a correlation matrix
"""

from typing import Optional
import numpy as np

from .waves_radon import WavesRadon
from .waves_radon import WavesSinogram
from .correlation_sinogram import CorrelationSinogram
from .correlation_image import CorrelationImage
from ..waves_exceptions import NoRadonTransformError


class CorrelationRadon(WavesRadon):
    """
    This class extends class WavesRadon and is used for radon transform computed from a centered
    correlation matrix
    It is very important that circle=True has been chosen to compute radon matrix since we read
    values in meters from the axis of the sinogram
    """

    def __init__(self, image: CorrelationImage, mean_filter_kernel_size,
                 directions_step: float = 1., weighted: bool = False):
        super().__init__(image, directions_step, weighted)
        self._mean_filter_kernel_size = mean_filter_kernel_size
        self._sinogram_maximum_variance = None

    def get_sinogram(self, direction: float) -> CorrelationSinogram:
        if self._radon_transform is None:
            raise NoRadonTransformError()
        return CorrelationSinogram(self._radon_transform.values_for(direction),
                                   self.sampling_frequency,
                                   self._mean_filter_kernel_size)

    @property
    def sinogram_maximum_variance(self) -> WavesSinogram:
        if self._sinogram_maximum_variance is None:
            self._sinogram_maximum_variance = self.get_sinogram_maximum_variance()
        return self._sinogram_maximum_variance

    def get_sinogram_maximum_variance(self,
                                      directions: Optional[np.ndarray] = None) -> (
    WavesSinogram, float):
        directions = self.directions if directions is None else directions
        maximum_variance = None
        sinogram_maximum_variance = None
        for result_index, sinogram_index in enumerate(directions):
            sinogram = self.sinograms[sinogram_index]
            if not maximum_variance:
                maximum_variance = sinogram.variance
                sinogram_maximum_variance = sinogram
            elif maximum_variance < sinogram.variance:
                maximum_variance = sinogram.variance
                sinogram_maximum_variance = sinogram
        return (sinogram_maximum_variance, directions[result_index])
