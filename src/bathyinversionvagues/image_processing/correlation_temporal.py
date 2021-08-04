import numpy as np
import scipy
from scipy.interpolate import interp1d


class CorrelationTemporal():
    def __init__(self, angle, angles, distances, celerity, correlation_matrix,
                 time_interpolation_resolution,
                 low_frequency_ratio, high_frequency_ratio, min_peaks_distance):
        self._angle = angle
        self._angles = angles
        self._distances = distances
        self._celerity = celerity
        self._correlation_matrix = correlation_matrix
        self._time_interpolation_resolution = time_interpolation_resolution
        self._low_frequency_ratio = low_frequency_ratio
        self._high_frequency_ratio = high_frequency_ratio
        self._min_peaks_distance = min_peaks_distance

        self._signal = None
        self._tuned_signal = None
        self._period = None
        self._max_peaks = None

    @property
    def signal(self) -> np.array:
        if self._signal is None:
            self._signal = self.compute_signal()
        return self._signal

    @property
    def tuned_signal(self) -> np.array:
        if self._tuned_signal is None:
            self._tuned_signal = self.compute_tuned_signal()
        return self._tuned_signal

    @property
    def max_peaks(self) -> np.array:
        if self._max_peaks is None:
            self._max_peaks, _ = scipy.signal.find_peaks(self.tuned_signal,
                                                         distance=self._min_peaks_distance)
        return self._max_peaks

    @property
    def period(self) -> float:
        if self._period is None:
            self._period = self.compute_period()
        return self._period

    def compute_signal(self):
        D = np.cos(np.radians(self._angle - self._angles.T.flatten())) * self._distances.flatten()
        time = D / self._celerity
        time_unique, index_unique = np.unique(time, return_index=True)
        index_unique_sorted = np.argsort(time_unique)
        time_unique_sorted = time_unique[index_unique_sorted]
        timevec = np.arange(np.min(time_unique_sorted), np.max(time_unique_sorted),
                            self._time_interpolation_resolution)
        corr_unique_sorted = self._correlation_matrix.T.flatten()[index_unique[index_unique_sorted]]
        interpolation = interp1d(time_unique_sorted, corr_unique_sorted)
        signal = interpolation(timevec)
        return signal

    def compute_tuned_signal(self):
        low_frequency = self._low_frequency_ratio * self._time_interpolation_resolution
        high_frequency = self._high_frequency_ratio * self._time_interpolation_resolution
        sos_filter = scipy.signal.butter(1, (2 * low_frequency, 2 * high_frequency),
                                         btype='bandpass', output='sos')
        tuned_signal = scipy.signal.sosfiltfilt(sos_filter, self.signal)
        return tuned_signal

    def compute_period(self):
        self._max_peaks, properties_max = scipy.signal.find_peaks(self.tuned_signal,
                                                                  distance=self._min_peaks_distance)
        period = np.mean(np.diff(self.max_peaks))
        return period
