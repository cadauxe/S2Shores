# -*- coding: utf-8 -*-
"""
module -- Class encapsulating operations on sinograms comming from a radon transform of a
correlation matrix
"""

import numpy as np

from .waves_sinogram import WavesSinogram


class CorrelationSinogram(WavesSinogram):
    """
    This class extends class WavesSinogram and is used for sinograms computed from radon matrix a
    centered correlation matrix
    It is very important that circle=True has been chosen to compute radon matrix since we read
    values in meters from the axis of the sinogram
    """

    def __init__(self, sinogram, sampling_frequency, mean_filter_kernel_size):
        super().__init__(sinogram, sampling_frequency)
        self._tuned_sinogram = None
        self._mean_filter_kernel_size = mean_filter_kernel_size

    @property
    def tuned_sinogram(self) -> np.array:
        if self._tuned_sinogram is None:
            # FIXME : tuned_sinogram should not use a parameter
            # Find a better way than median filter to tune sinogram, may be spline interpolation
            self._tuned_sinogram = self.filter_mean(self._mean_filter_kernel_size)
        return self._tuned_sinogram
