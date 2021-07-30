from .waves_radon import WavesRadon
from .waves_radon import WavesSinogram
from .correlation_sinogram import CorrelationSinogram
from .correlation_image import CorrelationImage
from ..waves_exceptions import NoRadonTransformError
from typing import Optional
import numpy as np


class CorrelationRadon(WavesRadon):
    """
    This class extends class WavesSinogram and is used for sinograms computed from radon matrix a centered correlation matrix
    It is very important that circle=True has been chosen to compute radon matrix since we read values in meters from the axis of the sinogram
    """

    def __init__(self, image: CorrelationImage, spatial_resolution, time_resolution, temporal_lag,
                 mean_filter_kernel_size, directions_step: float = 1., weighted: bool = False):
        super().__init__(image, directions_step, weighted)
        self._spatial_resolution = spatial_resolution
        self._time_resolution = time_resolution
        self._temporal_lag = temporal_lag
        self._mean_filter_kernel_size = mean_filter_kernel_size

    @property
    def spatial_resolution(self):
        return self._spatial_resolution

    @property
    def time_resolution(self):
        return self._time_resolution

    @property
    def temporal_lag(self):
        return self._temporal_lag

    def get_sinogram(self, direction: float) -> CorrelationSinogram:
        if self._radon_transform is None:
            raise NoRadonTransformError()
        return CorrelationSinogram(self._radon_transform.values_for(direction), self.sampling_frequency,
                                   self.spatial_resolution, self.time_resolution, self.temporal_lag,
                                   self._mean_filter_kernel_size)

    def get_sinogram_maximum_variance(self, directions: Optional[np.ndarray] = None) -> WavesSinogram:
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
        return sinogram_maximum_variance, directions[result_index]
