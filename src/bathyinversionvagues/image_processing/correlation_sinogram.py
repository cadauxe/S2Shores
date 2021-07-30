from .waves_sinogram import WavesSinogram
import numpy as np
from .shoresutils import filter_mean


class CorrelationSinogram(WavesSinogram):
    """
    This class extends class WavesSinogram and is used for sinograms computed from radon matrix a centered correlation matrix
    It is very important that circle=True has been chosen to compute radon matrix since we read values in meters from the axis of the sinogram
    """

    def __init__(self, sinogram, sampling_frequency, spatial_resolution, time_resolution, temporal_lag,
                 mean_filter_kernel_size):
        super().__init__(sinogram, sampling_frequency)
        self._wave_length = None
        self._celerity = None
        self._tuned_sinogram = None
        self._spatial_resolution = spatial_resolution
        self.max_distance = (sinogram.size * spatial_resolution) / 2
        self.x = np.linspace(-self.max_distance / 2, self.max_distance / 2, sinogram.size)
        self._time_resolution = time_resolution
        self._temporal_lag = temporal_lag
        self._mean_filter_kernel_size = mean_filter_kernel_size

    @property
    def wave_length(self) -> float:
        if self._wave_length is None:
            self._wave_length = self.compute_wave_length()
        return self._wave_length

    @property
    def celerity(self) -> float:
        if self._celerity is None:
            self._celerity = self.compute_celerity()
        return self._celerity

    @property
    def tuned_sinogram(self) -> np.array:
        if self._tuned_sinogram is None:
            self._tuned_sinogram = self.compute_tuned_sinogram()
        return self._tuned_sinogram

    def compute_wave_length(self):
        sign = np.sign(self.tuned_sinogram)
        diff = np.diff(sign)
        zeros = np.where(diff != 0)[0]
        wave_length = 2 * np.mean(np.diff(zeros))
        return wave_length

    def compute_celerity(self):
        size_sinogram = len(self.tuned_sinogram)
        m1 = max(int(size_sinogram / 2 - self.wave_length / 2), 0)
        m2 = min(int(size_sinogram / 2 + self.wave_length / 2), size_sinogram)
        argmax = np.argmax(self.tuned_sinogram[m1:m2])
        rhomx = self._spatial_resolution * np.abs(argmax + m1 - size_sinogram / 2)
        t = self._time_resolution * self._temporal_lag
        celerity = np.abs(rhomx / t)
        return celerity

    def compute_tuned_sinogram(self):
        # FIXME : tuned_sinogram should not use a parameter
        # Find a better way than median filter to tune sinogram, may be spline interpolation
        array = np.ndarray.flatten(self.sinogram)
        sinogram_max_var_tuned = filter_mean(array, self._mean_filter_kernel_size)
        return sinogram_max_var_tuned
