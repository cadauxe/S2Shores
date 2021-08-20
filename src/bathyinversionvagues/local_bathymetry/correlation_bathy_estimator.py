# -*- coding: utf-8 -*-
"""
Abstract Class offering a common template for temporal correlation method and spatial correlation
method

@author: erwinbergsma
         gregoirethoumyre
         degoulromain
"""
from abc import abstractmethod, abstractproperty
from typing import Optional, List  # @NoMove

from munch import Munch
from scipy.interpolate import interp1d
from scipy.signal import butter, find_peaks, sosfiltfilt

import numpy as np

from ..image_processing.correlation_image import CorrelationImage
from ..image_processing.waves_image import WavesImage
from ..image_processing.waves_radon import WavesRadon
from .local_bathy_estimator import LocalBathyEstimator


class CorrelationBathyEstimator(LocalBathyEstimator):
    """
    Class offering a framework for bathymetry computation based on correlation
    """

    def __init__(self, images_sequence: List[WavesImage], global_estimator,
                 selected_directions: Optional[np.ndarray] = None) -> None:
        """ constructor
        :param images_sequence: sequence of image used to compute bathymetry
        :param global_estimator: global estimator
        :param selected_directions: selected_directions: the set of directions onto which the
        sinogram must be computed
        """

        super().__init__(images_sequence, global_estimator, selected_directions)
        self._correlation_matrix: np.ndarray = None
        self._correlation_image: CorrelationImage = None
        self.radon_transform: Optional[WavesRadon] = None
        self._angles: np.ndarray = None
        self._distances: np.ndarray = None

    def run(self) -> None:
        """ Run the local bathy estimator using correlation method
        """
        try:
            self.create_radon_transform()
            # It is very important that circle=True has been chosen to compute radon matrix since
            # we read values in meters from the axis of the sinogram
            self.radon_transform.compute()
            sinogram_max_var, direction_propagation = \
                self.radon_transform.get_sinogram_maximum_variance()
            # TODO: Find a better way to tune sinogram, may be spline interpolation
            tuned_sinogram_max_var = sinogram_max_var.filter_mean(
                self._parameters.TUNING.MEAN_FILTER_KERNEL_SIZE_SINOGRAM)

            # TODO: gather following 2 calls in a single method returning a consistent wave_length
            # celerity pair
            wave_length = self.compute_wave_length(tuned_sinogram_max_var)
            celerity = self.compute_celerity(tuned_sinogram_max_var, wave_length)
            temporal_signal = self.temporal_reconstruction(direction_propagation, celerity)
            temporal_signal_filtered = self.temporal_reconstruction_tuning(temporal_signal)
            period = self.compute_period(temporal_signal_filtered)
            waves_field_estimation = self.create_waves_field_estimation(direction_propagation,
                                                                        wave_length)
            waves_field_estimation.period = period
            waves_field_estimation.celerity = celerity
            self.store_estimation(waves_field_estimation)
        except Exception as excp:
            print(f'Bathymetry computation failed: {str(excp)}')

    @abstractproperty
    def _parameters(self) -> Munch:
        """
        :return: munchified parameters
        """
        # FIXME: Why not using parameters from global bathy estimatror (this is
        # the general principle)

    @abstractproperty
    def positions_x(self) -> np.ndarray:
        """
        :return: ndarray of x positions
        """

    @abstractproperty
    def positions_y(self) -> np.ndarray:
        """
        :return: ndarray of y positions
        """

    @abstractmethod
    def get_correlation_matrix(self) -> np.ndarray:
        """
        :return: correlation matrix
        """

    @abstractmethod
    def get_correlation_image(self) -> CorrelationImage:
        """
        :return: correlation image
        """

    def get_angles(self) -> np.ndarray:
        """
        Angles are in degrees
        :return: the angles between all points selected to compute correlation
        """
        xrawipool_ik_dist = np.tile(self.positions_x, (len(self.positions_x), 1)) - np.tile(
            self.positions_x.T, (1, len(self.positions_x)))
        yrawipool_ik_dist = np.tile(self.positions_y, (len(self.positions_y), 1)) - np.tile(
            self.positions_y.T, (1, len(self.positions_y)))
        return np.arctan2(xrawipool_ik_dist, yrawipool_ik_dist).T * 180 / np.pi

    def get_distances(self) -> np.ndarray:
        """
        Distances between positions x and positions y
        Be aware these distances are not in meter and have to be multiplied by spatial resolution
        :return: the distances between all points selected to compute correlation
        """
        return np.sqrt(np.square((self.positions_x - self.positions_x.T)) + np.square(
            (self.positions_y - self.positions_y.T)))

    @property
    def correlation_image(self) -> CorrelationImage:
        """
        :return: correlation image used to perform radon transformation
        """
        if self._correlation_image is None:
            self._correlation_image = self.get_correlation_image()
        return self._correlation_image

    @property
    def correlation_matrix(self) -> np.ndarray:
        """
        Be aware this matrix is projected before radon transformation in temporal correlation case
        :return: correlation matrix used for temporal reconstruction
        """
        if self._correlation_matrix is None:
            self._correlation_matrix = self.get_correlation_matrix()
        return self._correlation_matrix

    @property
    def angles(self) -> np.ndarray:
        """
        :return: angles in radian
        """
        if self._angles is None:
            self._angles = self.get_angles()
        return self._angles

    @property
    def distances(self) -> np.ndarray:
        """
        :return: distances
        """
        if self._distances is None:
            self._distances = self.get_distances()
        return self._distances

    def create_radon_transform(self) -> None:
        """
        Creation of the radon transform. Note that correlation_image must have resolution of 1 meter
        :return:
        """
        self.radon_transform = WavesRadon(self.correlation_image)

    def compute_wave_length(self, sinogram: np.ndarray) -> float:
        """
        Wave length computation
        :param sinogram: sinogram on which wave length is computed
        :return: wave length in meter
        """
        # TODO: move following 4 lines in a generic signal processing function
        sign = np.sign(sinogram)
        diff = np.diff(sign)
        zeros = np.where(diff != 0)[0]
        difference = 2 * np.mean(np.diff(zeros))

        wave_length = difference * self._parameters.RESOLUTION.SPATIAL
        return wave_length

    def compute_celerity(self, sinogram: np.ndarray, wave_length: float) -> float:
        """
        Celerity computation
        :param sinogram: sinogram on which celerity is computed
        :param wave_length: wave length of the sinogram
        :return: celerity in meter/second
        """
        # TODO: move following 5 lines in a generic signal processing function
        size_sinogram = len(sinogram)
        left_limit = max(int(size_sinogram / 2 - wave_length / 2), 0)
        right_limit = min(int(size_sinogram / 2 + wave_length / 2), size_sinogram)
        argmax = np.argmax(sinogram[left_limit:right_limit])
        difference_max = np.abs(argmax + left_limit - size_sinogram / 2)

        rhomx = self._parameters.RESOLUTION.SPATIAL * difference_max
        duration = self._parameters.RESOLUTION.TEMPORAL * self._parameters.TEMPORAL_LAG
        celerity = np.abs(rhomx / duration)
        return celerity

    def temporal_reconstruction(self, direction_propagation, celerity):
        """
        Temporal reconstruction of the correlation signal following propagation direction
        :param direction_propagation: propagation angles in degrees
        :param celerity: celerity in meter/second
        :return: correlation temporal signal
        """
        distances = np.cos(
            np.radians(
                direction_propagation - self.angles.T.flatten())) * self.distances.flatten() * self._parameters.RESOLUTION.SPATIAL
        time = distances / celerity
        time_unique, index_unique = np.unique(time, return_index=True)
        index_unique_sorted = np.argsort(time_unique)
        time_unique_sorted = time_unique[index_unique_sorted]
        timevec = np.arange(np.min(time_unique_sorted), np.max(time_unique_sorted),
                            self._parameters.RESOLUTION.TIME_INTERPOLATION)
        corr_unique_sorted = self.correlation_matrix.T.flatten()[index_unique[index_unique_sorted]]
        interpolation = interp1d(time_unique_sorted, corr_unique_sorted)
        return interpolation(timevec)

    def temporal_reconstruction_tuning(self, temporal_signal):
        """
        Tuning of temporal signal
        :param temporal_signal: temporal signal
        :return: tuned temporal signal
        """
        low_frequency = self._parameters.TUNING.LOW_FREQUENCY_RATIO_TEMPORAL_RECONSTRUCTION * \
            self._parameters.RESOLUTION.TIME_INTERPOLATION
        high_frequency = self._parameters.TUNING.HIGH_FREQUENCY_RATIO_TEMPORAL_RECONSTRUCTION * \
            self._parameters.RESOLUTION.TIME_INTERPOLATION
        sos_filter = butter(1, (2 * low_frequency, 2 * high_frequency),
                            btype='bandpass', output='sos')
        temporal_signal_filtered = sosfiltfilt(sos_filter, temporal_signal)
        return temporal_signal_filtered

    def compute_period(self, temporal_signal_filtered):
        """
        Period computation
        :param temporal_signal_filtered: temporal signal filtered
        :return: period in second
        """
        peaks_max, _ = find_peaks(temporal_signal_filtered,
                                  distance=self._parameters.TUNING.MIN_PEAKS_DISTANCE_PERIOD)
        period = np.mean(np.diff(peaks_max))
        return period
