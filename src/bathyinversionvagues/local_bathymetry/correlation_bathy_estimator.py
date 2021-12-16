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
import warnings

import numpy as np

from ..bathy_physics import wavelength_offshore
from ..generic_utils.image_filters import detrend, clipping
from ..generic_utils.signal_filters import filter_mean, remove_median
from ..generic_utils.signal_utils import find_period_from_peaks, find_period_from_zeros
from ..image_processing.waves_image import WavesImage, ImageProcessingFilters
from ..image_processing.waves_radon import WavesRadon, linear_directions
from ..image_processing.waves_sinogram import SignalProcessingFilters
from ..waves_exceptions import WavesEstimationError

from .correlation_waves_field_estimation import CorrelationWavesFieldEstimation
from .local_bathy_estimator import LocalBathyEstimator
from .waves_fields_estimations import WavesFieldsEstimations

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
        if self.selected_directions is None:
            self.selected_directions = linear_directions(-180., 0., 1.)
        # Processing attributes
        self._correlation_matrix: Optional[np.ndarray] = None
        self._correlation_image: Optional[WavesImage] = None
        self.radon_transform: Optional[WavesRadon] = None
        self._angles: Optional[np.ndarray] = None
        self._distances: Optional[np.ndarray] = None
        # Filters
        self.correlation_image_filters: ImageProcessingFilters = [(detrend, []), (
            clipping, [self.local_estimator_params['TUNING']['RATIO_SIZE_CORRELATION']])]
        self.radon_image_filters: SignalProcessingFilters = [
            (remove_median,
             [self.local_estimator_params['TUNING']['MEDIAN_FILTER_KERNEL_RATIO_SINOGRAM']]),
            (filter_mean, [self.local_estimator_params['TUNING']['MEAN_FILTER_KERNEL_SIZE_SINOGRAM']])]
        if self.local_estimator_params['TEMPORAL_LAG'] >= len(self._sequential_delta_times):
            raise WaveEstimationError(
                'The chosen number of lag frames is bigger than the number of available frames')
        self.propagation_duration = np.sum(
            self._sequential_delta_times[:self.local_estimator_params['TEMPORAL_LAG']])
        self._metrics['propagation_duration'] = self.propagation_duration


    def run(self) -> None:
        """ Run the local bathy estimator using correlation method
        """
        try:
            self.correlation_image.apply_filters(self.correlation_image_filters)
            radon_transform = WavesRadon(self.correlation_image, self.selected_directions)
            filtered_radon = radon_transform.apply_filters(self.radon_image_filters)
            direction_propagation, variances = \
                filtered_radon.get_direction_maximum_variance()
            sinogram_max_var = radon_transform[direction_propagation]
            sinogram_max_var_values = sinogram_max_var.values
            self._metrics['sinogram_max_var'] = radon_transform.values
            wave_length = self.compute_wave_length(sinogram_max_var_values)
            celerities = self.compute_celerities(
                sinogram_max_var_values, wave_length, self.local_estimator_params['HOPS_NUMBER'])
            temporal_signals = []
            periods = []
            arg_peaks_max = []
            for celerity in celerities:
                temporal_signal = self.temporal_reconstruction(celerity, direction_propagation)
                try:
                    temporal_signal = self.temporal_reconstruction_tuning(temporal_signal)
                except ValueError:
                    warnings.warn('Temporal signal is too short to be filtered')
                temporal_signals.append(temporal_signal)
                period, arg_peak_max = find_period_from_peaks(
                    temporal_signal, min_period=self.global_estimator.waves_period_min)
                periods.append(period)
                arg_peaks_max.append(arg_peak_max)

            celerities_from_periods = [wave_length / period for period in periods]
            errors_celerities = np.abs(celerities_from_periods - celerities)
            index_min = np.nanargmin(errors_celerities)
            celerity = celerities[index_min]
            period = periods[index_min]
            waves_field_estimation = cast(CorrelationWavesFieldEstimation,
                                          self.create_waves_field_estimation(direction_propagation,
                                                                             wave_length))
            waves_field_estimation.period = period
            waves_field_estimation.celerity = celerity
            self.store_estimation(waves_field_estimation)
            print(waves_field_estimation)

            if self.debug_sample:
                self._metrics['radon_transform'] = radon_transform
                self._metrics['variances'] = variances
                self._metrics['sinogram_max_var'] = sinogram_max_var_values
                # TODO: use objects in debug
                self._metrics['temporal_signals'] = temporal_signals
                self._metrics['arg_peaks_max'] = arg_peaks_max
                self._metrics['periods'] = periods
                self._metrics['celerities'] = celerities
                self._metrics['celerities_from_periods'] = celerities_from_periods
        except Exception as excp:
            print(f'Bathymetry computation failed: {str(excp)}')

    def sort_waves_fields(self) -> None:
        """ Sort the waves fields estimations based on their energy max.
        """
        # FIXME: (ROMAIN) decide if some specific sorting is needed

    @property
    @abstractmethod
    def sampling_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """ :return: tuples (x,y) of nd.array positions
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

    def compute_wave_length(self, sinogram: np.ndarray) -> float:
        """ Wave length computation (in meter)
        :param sinogram : sinogram used to compute wave length
        :returns: wave length
        """
        min_wavelength = wavelength_offshore(self.global_estimator.waves_period_min, self.gravity)
        period, wave_length_zeros = find_period_from_zeros(sinogram,
                                                           int(min_wavelength / self.spatial_resolution))
        wave_length = period * self.spatial_resolution

        if self.debug_sample:
            self._metrics['wave_length_zeros'] = wave_length_zeros
        return wave_length

    def compute_celerities(self, sinogram: np.ndarray, wave_length: float, nb_hops: int) -> np.ndarray:
        """ Propagated distance computation (in meter)
        - 1) sinogram maximum is determined on interval [-wave_length, wave_length]
        - 2) nb_max_hops propagated distances are computed from the position of the maximum using following formula :
            [dx , dx + wave_length, ..., dx + (nb_max_hops)*wave_length] (an adaptation to negative values is also made if needed)
        :param sinogram: sinogram having maximum variance
        :param wave_length: wave_length computed on sinogram
        :param nb_hops: number of propagated distances computed
        :returns: np.ndarray of size nb_hops containing computed celerities
        """
        x = np.arange(-(len(sinogram) // 2), len(sinogram) // 2 + 1)
        interval = np.logical_and(x * self.spatial_resolution > -wave_length, x * self.spatial_resolution < wave_length)
        self._metrics['interval'] = interval
        peaks, _ = find_peaks(sinogram)
        peaks = peaks[interval[peaks]]
        max_indice = np.argmax(sinogram[peaks])
        dx = x[peaks[max_indice]]
        if dx > 0:
            dephasings = dx * self.spatial_resolution + wave_length * np.arange(nb_hops)
            max_indices = np.array(peaks[max_indice] +
                                   np.arange(nb_hops) * wave_length / self.spatial_resolution, dtype=int)
        else:
            dephasings = dx * self.spatial_resolution - wave_length * np.arange(nb_hops)
            dephasings = np.abs(dephasings)
            max_indices = np.array(peaks[max_indice] -
                                   np.arange(nb_hops) * wave_length / self.spatial_resolution, dtype=int)
        max_indices = max_indices[np.logical_and(max_indices > 0, max_indices < len(sinogram))]

        celerities = dephasings / self.propagation_duration

        if self.debug_sample:
            self._metrics['x'] = x
            self._metrics['max_indices'] = max_indices
            self._metrics['dephasings'] = dephasings
            self._metrics['propagation_duration'] = self.propagation_duration
        return celerities

    def temporal_reconstruction(self, celerity: float, direction_propagation: float) -> np.ndarray:
        """ Temporal reconstruction of the correlation signal following propagation direction
        :param celerity : computed celerity in meter/second
        :param direction_propagation: angle of direction propagation in degrees
        :returns: temporal reconstruction of the signal
        """
        distances = np.cos(np.radians(direction_propagation - self.angles.T.flatten())) * \
            self.distances.flatten() * self.spatial_resolution
        time = distances / celerity
        time_unique, index_unique = np.unique(time, return_index=True)
        index_unique_sorted = np.argsort(time_unique)
        time_unique_sorted = time_unique[index_unique_sorted]
        timevec = np.arange(np.min(time_unique_sorted), np.max(time_unique_sorted),
                            self.local_estimator_params['RESOLUTION']['TIME_INTERPOLATION'])
        corr_unique_sorted = self.correlation_matrix.T.flatten()[
            index_unique[index_unique_sorted]]
        interpolation = interp1d(time_unique_sorted, corr_unique_sorted)
        temporal_signal = interpolation(timevec)
        return temporal_signal

    def temporal_reconstruction_tuning(self, temporal_signal: np.ndarray) -> np.ndarray:
        """ Tuning of temporal signal
        :param temporal_signal : temporal signal to be tuned
        :raises ValueError: when the signal is too short to be filtered
        :returns: tuned temporal signal
        """
        low_frequency = \
            self.global_estimator.waves_period_max * \
            self.local_estimator_params['RESOLUTION']['TIME_INTERPOLATION']
        high_frequency = \
            self.global_estimator.waves_period_min \
            * self.local_estimator_params['RESOLUTION']['TIME_INTERPOLATION']
        sos_filter = butter(1, (2 * low_frequency, 2 * high_frequency),
                            btype='bandpass', output='sos')
        # Formula found on :
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html
        padlen = 3 * (2 * len(sos_filter) + 1 - min((sos_filter[:, 2] == 0).sum(),
                                                    (sos_filter[:, 5] == 0).sum()))
        if not len(temporal_signal) > padlen:
            raise ValueError('Temporal signal is too short to be filtered')
        return sosfiltfilt(sos_filter, temporal_signal)

