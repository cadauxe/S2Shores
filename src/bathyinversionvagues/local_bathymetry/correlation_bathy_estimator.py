# -*- coding: utf-8 -*-
""" Abstract Class offering a common template for temporal correlation method and spatial
correlation method

:author: Degoul Romain
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 18/06/2021
"""
from abc import abstractmethod
from typing import Optional, List, Tuple, TYPE_CHECKING, cast  # @NoMove

from scipy.interpolate import interp1d
from scipy.signal import butter, find_peaks, sosfiltfilt

import numpy as np

from ..generic_utils.image_filters import detrend, clipping
from ..generic_utils.signal_filters import filter_mean, remove_median
from ..generic_utils.signal_utils import find_period
from ..image_processing.waves_image import WavesImage, ImageProcessingFilters
from ..image_processing.waves_radon import WavesRadon
from ..image_processing.waves_sinogram import SignalProcessingFilters

from .correlation_waves_field_estimation import CorrelationWavesFieldEstimation
from .local_bathy_estimator import LocalBathyEstimator
from .waves_fields_estimations import WavesFieldsEstimations
from ..waves_exceptions import WavesEstimationError

if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport


class CorrelationBathyEstimator(LocalBathyEstimator):
    """ Class offering a framework for bathymetry computation based on correlation
    """
    waves_field_estimation_cls = CorrelationWavesFieldEstimation

    def __init__(self, images_sequence: List[WavesImage], global_estimator: 'BathyEstimator',
                 waves_fields_estimations: WavesFieldsEstimations,
                 selected_directions: Optional[np.ndarray] = None) -> None:
        super().__init__(images_sequence, global_estimator,
                         waves_fields_estimations, selected_directions)
        # Processing attributes
        self._correlation_matrix: Optional[np.ndarray] = None
        self._correlation_image: Optional[WavesImage] = None
        self.radon_transform: Optional[WavesRadon] = None
        self._angles: Optional[np.ndarray] = None
        self._distances: Optional[np.ndarray] = None
        # Filters
        self.correlation_image_filters: ImageProcessingFilters = [(detrend, []), (
            clipping, [self.local_estimator_params.TUNING.RATIO_SIZE_CORRELATION])]
        self.radon_image_filters: SignalProcessingFilters = [
            (remove_median,
             [self.local_estimator_params.TUNING.MEDIAN_FILTER_KERNEL_RATIO_SINOGRAM]),
            (filter_mean, [self.local_estimator_params.TUNING.MEAN_FILTER_KERNEL_SIZE_SINOGRAM])]
        if self.local_estimator_params.TEMPORAL_LAG >= len(self._sequential_delta_times):
            raise WaveEstimationError(
                'There are enough frames compared to chosen number of lag frames')
        self.delta_time = np.sum(
            self._sequential_delta_times[:self.local_estimator_params.TEMPORAL_LAG])
        self._metrics['delta_time'] = self.delta_time

    def run(self) -> None:
        """ Run the local bathy estimator using correlation method
        """
        try:
            self.correlation_image.apply_filters(self.correlation_image_filters)
            # TODO: remove this attribute.
            self.radon_transform = WavesRadon(self.correlation_image, self.selected_directions)
            # FIXME: store filtered_sinograms into metrics (was previously displaed)
            filtered_sinograms = self.radon_transform.apply_filters(self.radon_image_filters)
            direction_propagation, variances = \
                filtered_sinograms.get_direction_maximum_variance()
            sinogram_max_var = filtered_sinograms[direction_propagation]
            sinogram_max_var_values = sinogram_max_var.values
            self._metrics['sinogram_max_var'] = sinogram_max_var_values
            wave_length = self.compute_wave_length(
                sinogram_max_var_values, min_period=self.local_estimator_params.TUNING.MINIMUM_WAVE_LENGTH)
            celerity = self.compute_celerity(sinogram_max_var_values, wave_length)
            temporal_signal = self.temporal_reconstruction(celerity, direction_propagation)
            temporal_signal = self.temporal_reconstruction_tuning(temporal_signal)
            period = self.compute_period(temporal_signal)

            waves_field_estimation = cast(CorrelationWavesFieldEstimation,
                                          self.create_waves_field_estimation(direction_propagation,
                                                                             wave_length))
            waves_field_estimation.period = period
            waves_field_estimation.celerity = celerity
            self.store_estimation(waves_field_estimation)

            if self.debug_sample:
                self._metrics['variances'] = variances
                self._metrics['sinogram_max_var'] = sinogram_max_var.sinogram.flatten()
                self._metrics['temporal_signal'] = temporal_signal
        except Exception as excp:
            print(f'Bathymetry computation failed: {str(excp)}')

    def sort_waves_fields(self) -> None:
        """ Sort the waves fields estimations based on their energy max.
        """
        # FIXME: (ROMAIN) decide if some specific sorting is needed

    @property
    @abstractmethod
    def sampling_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """ :return: tuple (x,y) of nd.array positions
        """

    @abstractmethod
    def get_correlation_matrix(self) -> np.ndarray:
        """ :return: correlation matrix
        """

    def get_correlation_image(self) -> WavesImage:
        """ :return: correlation image
        """
        return WavesImage(self.correlation_matrix, self.spatial_resolution)

    @property
    def preprocessing_filters(self) -> ImageProcessingFilters:
        """ :returns: A list of functions together with their parameters to be applied
        sequentially to all the images of the sequence before subsequent bathymetry estimation.
        """
        preprocessing_filters: ImageProcessingFilters = []
        return preprocessing_filters

    def get_angles(self) -> np.ndarray:
        """ Get the angles between all points selected to compute correlation
        :return: Angles (in degrees)
        """
        xrawipool_ik_dist = \
            np.tile(self.sampling_positions[0], (len(self.sampling_positions[0]), 1)) - \
            np.tile(self.sampling_positions[0].T, (1, len(self.sampling_positions[0])))
        yrawipool_ik_dist = \
            np.tile(self.sampling_positions[1], (len(self.sampling_positions[1]), 1)) - \
            np.tile(self.sampling_positions[1].T, (1, len(self.sampling_positions[1])))
        return np.arctan2(yrawipool_ik_dist, xrawipool_ik_dist) * 180 / np.pi

    def get_distances(self) -> np.ndarray:
        """ Distances between positions x and positions y
        Be aware these distances are not in meter and have to be multiplied by spatial resolution

        :return: the distances between all points selected to compute correlation
        """
        return np.sqrt(
            np.square((self.sampling_positions[0] - self.sampling_positions[0].T)) +
            np.square((self.sampling_positions[1] - self.sampling_positions[1].T)))

    @property
    def correlation_image(self) -> WavesImage:
        """ :return: correlation image used to perform radon transformation
        """
        if self._correlation_image is None:
            self._correlation_image = self.get_correlation_image()
        return self._correlation_image

    @property
    def correlation_matrix(self) -> np.ndarray:
        """ Be aware this matrix is projected before radon transformation in temporal correlation
        case

        :return: correlation matrix used for temporal reconstruction
        """
        if self._correlation_matrix is None:
            self._correlation_matrix = self.get_correlation_matrix()
        return self._correlation_matrix

    @property
    def angles(self) -> np.ndarray:
        """ :return: angles in radian
        """
        if self._angles is None:
            self._angles = self.get_angles()
        return self._angles

    @property
    def distances(self) -> np.ndarray:
        """ :return: distances
        """
        if self._distances is None:
            self._distances = self.get_distances()
        return self._distances

    def compute_wave_length(self, sinogram: np.ndarray, min_period: int) -> float:
        """ Wave length computation (in meter)
        """
        period, wave_length_zeros = find_period(sinogram, min_period)
        wave_length = period * self.spatial_resolution

        if self.debug_sample:
            self._metrics['wave_length_zeros'] = wave_length_zeros
        return wave_length

    def find_propagated_distance(self, signal: np.ndarray, wave_length: float) -> float:
        t_offshore = np.sqrt(wave_length * 2 * np.pi / self.gravity)
        self._metrics['t_offshore'] = t_offshore
        nb_l = int(self.delta_time // t_offshore)
        x = np.arange(-(len(signal) // 2), len(signal) // 2 + 1)
        interval = np.ones(x.shape, dtype=bool)
        self._metrics['interval'] = interval
        signal[np.logical_not(interval)] = 0
        max_indice = np.argmax(signal)
        self._metrics['max_indice'] = max_indice
        dx = abs(x[np.argmax(signal)]) * self.spatial_resolution
        dephasing = nb_l * wave_length + dx
        self._metrics['dephasing'] = dephasing
        self._metrics['dx'] = dx
        self._metrics['nb_l'] = nb_l
        return dephasing

    def compute_celerity(self, sinogram: np.ndarray, wave_length: float) -> float:
        """ Celerity computation (in meter/second)
        """
        dephasing = self.find_propagated_distance(sinogram, wave_length)
        celerity = np.abs(dephasing / self.delta_time)
        return celerity

    def temporal_reconstruction(self, celerity: float, direction_propagation: float) -> np.ndarray:
        """ Temporal reconstruction of the correlation signal following propagation direction
        """
        distances = np.cos(np.radians(direction_propagation - self.angles.T.flatten())) * \
            self.distances.flatten() * self.spatial_resolution
        time = distances / celerity
        time_unique, index_unique = np.unique(time, return_index=True)
        index_unique_sorted = np.argsort(time_unique)
        time_unique_sorted = time_unique[index_unique_sorted]
        timevec = np.arange(np.min(time_unique_sorted), np.max(time_unique_sorted),
                            self.local_estimator_params.RESOLUTION.TIME_INTERPOLATION)
        corr_unique_sorted = self.correlation_matrix.T.flatten()[
            index_unique[index_unique_sorted]]
        interpolation = interp1d(time_unique_sorted, corr_unique_sorted)
        temporal_signal = interpolation(timevec)
        return temporal_signal

    def temporal_reconstruction_tuning(self, temporal_signal: np.ndarray) -> np.ndarray:
        """ Tuning of temporal signal
        """
        low_frequency = \
            self.local_estimator_params.TUNING.LOW_FREQUENCY_RATIO_TEMPORAL_RECONSTRUCTION * \
            self.local_estimator_params.RESOLUTION.TIME_INTERPOLATION
        high_frequency = \
            self.local_estimator_params.TUNING.HIGH_FREQUENCY_RATIO_TEMPORAL_RECONSTRUCTION * \
            self.local_estimator_params.RESOLUTION.TIME_INTERPOLATION
        sos_filter = butter(1, (2 * low_frequency, 2 * high_frequency),
                            btype='bandpass', output='sos')
        return sosfiltfilt(sos_filter, temporal_signal)

    def compute_period(self, temporal_signal: np.ndarray) -> float:
        """Period computation (in second)
        """
        arg_peaks_max, _ = find_peaks(
            temporal_signal, distance=self.local_estimator_params.TUNING.MIN_PEAKS_DISTANCE_PERIOD)

        period = float(np.mean(np.diff(arg_peaks_max)))

        if self.debug_sample:
            self._metrics['arg_temporal_peaks_max'] = arg_peaks_max
        return period
