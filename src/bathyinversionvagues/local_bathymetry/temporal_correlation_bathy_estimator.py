# -*- coding: utf-8 -*-
""" Class performing bathymetry computation using temporal correlation method

:author: Degoul Romain
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 18/06/2021
"""
from typing import Optional, List, Tuple, TYPE_CHECKING, cast  # @NoMove

import pandas
from scipy.signal import find_peaks

import numpy as np

from ..bathy_physics import wavelength_offshore
from ..data_model.waves_fields_estimations import WavesFieldsEstimations
from ..generic_utils.image_filters import detrend, clipping
from ..generic_utils.image_utils import cross_correlation
from ..generic_utils.signal_filters import filter_mean, remove_median
from ..generic_utils.signal_utils import find_period_from_zeros
from ..image_processing.waves_image import WavesImage, ImageProcessingFilters
from ..image_processing.waves_radon import WavesRadon, linear_directions
from ..image_processing.waves_sinogram import SignalProcessingFilters
from ..waves_exceptions import WavesEstimationError

from .local_bathy_estimator import LocalBathyEstimator
from .temporal_correlation_waves_field_estimation import TemporalCorrelationWavesFieldEstimation


if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport


class TemporalCorrelationBathyEstimator(LocalBathyEstimator):
    """ Class performing temporal correlation to compute bathymetry
    """
    waves_field_estimation_cls = TemporalCorrelationWavesFieldEstimation

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
            (filter_mean,
             [self.local_estimator_params['TUNING']['MEAN_FILTER_KERNEL_SIZE_SINOGRAM']])]
        self.create_sequence_time_series()
        if self.local_estimator_params['TEMPORAL_LAG'] >= len(self._sequential_delta_times):
            raise WavesEstimationError(
                'The chosen number of lag frames is bigger than the number of available frames')

    def create_sequence_time_series(self) -> None:
        """ This function computes an np.array of time series.
        To do this random points are selected within the sequence of image and a temporal serie
        is included in the np.array for each selected point
        """
        percentage_points = self.local_estimator_params['PERCENTAGE_POINTS']
        if percentage_points < 0 or percentage_points > 100:
            raise ValueError('Percentage must be between 0 and 100')
        merge_array = np.dstack([image.pixels for image in self.images_sequence])
        shape_x, shape_y = self.images_sequence[0].pixels.shape
        time_series = np.reshape(merge_array, (shape_x * shape_y, -1))
        # A seed is used here to reproduce same results
        np.random.seed(0)
        nb_random_points = round(shape_x * shape_y * percentage_points / 100)
        random_indexes = np.random.randint(0, shape_x * shape_y, size=nb_random_points)
        positions_y, positions_x = np.meshgrid(np.linspace(1, shape_x, shape_x),
                                               np.linspace(1, shape_y, shape_y))

        sampling_positions_x = np.reshape(positions_x.flatten()[random_indexes], (1, -1))
        sampling_positions_y = np.reshape(positions_y.flatten()[random_indexes], (1, -1))
        self._sampling_positions = (sampling_positions_x, sampling_positions_y)
        self._time_series = time_series[random_indexes, :]

    def run(self) -> None:
        """ Run the local bathy estimator using correlation method
        """
        try:
            filtered_image = self.correlation_image.apply_filters(self.correlation_image_filters)
            self.correlation_image.pixels = filtered_image.pixels
            radon_transform = WavesRadon(self.correlation_image, self.selected_directions)
            filtered_radon = radon_transform.apply_filters(self.radon_image_filters)
            direction_propagation, variances = filtered_radon.get_direction_maximum_variance()
            sinogram_max_var = radon_transform[direction_propagation]
            sinogram_max_var_values = sinogram_max_var.values
            wave_length = self.compute_wave_length(sinogram_max_var_values)
            distances = self.compute_distances(
                sinogram_max_var_values,
                wave_length,
                nb_hops=self.local_estimator_params['HOPS_NUMBER'])

            propagation_duration = np.sum(
                self._sequential_delta_times[:self.local_estimator_params['TEMPORAL_LAG']])
            travelled_distance = self.compute_travelled_distance(
                distances, propagation_duration, wave_length)

            waves_field_estimation = cast(TemporalCorrelationWavesFieldEstimation,
                                          self.create_waves_field_estimation(direction_propagation,
                                                                             wave_length))
            waves_field_estimation.delta_time = propagation_duration
            waves_field_estimation.propagated_distance = travelled_distance
            self.store_estimation(waves_field_estimation)

            if self.debug_sample:
                self.metrics['radon_transform'] = radon_transform
                self.metrics['variances'] = variances
                self.metrics['sinogram_max_var'] = sinogram_max_var_values
                # TODO: use objects in debug
                self.metrics['propagation_duration'] = propagation_duration
                self.metrics['distances'] = distances
        except Exception as excp:
            print(f'Bathymetry computation failed: {str(excp)}')

    def sort_waves_fields(self) -> None:
        pass

    @property
    def sampling_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """ :return: tuple of sampling positions
        """
        return self._sampling_positions

    def get_correlation_matrix(self) -> np.ndarray:
        """Compute temporal correlation matrix
        """
        temporal_lag = self.local_estimator_params['TEMPORAL_LAG']
        return cross_correlation(self._time_series[:, temporal_lag:],
                                 self._time_series[:, :-temporal_lag])

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

    def compute_travelled_distance(self, distances: np.ndarray,
                                   propagation_duration: float, wave_length: float) -> float:
        """ This function selects the travelled_distance among entry distances according to
        the linerity coefficient closest to minimal value

        :param distances: all candidates for travelled distances
        :param propagation_duration: the duration used to propagate waves
        :param wave_length: wave_length of selected wave
        :returns: travelled distances by the selected wave during propagation_duration
        :raises ValueError: when there is no linearity coefficient between min and max values
        """
        celerities = np.abs(distances / propagation_duration)
        linearity_coefficients = (2 * np.pi * celerities**2) / (wave_length * self.gravity)
        possible_values = np.logical_and(
            linearity_coefficients > self.global_estimator.waves_linearity_min,
            linearity_coefficients < self.global_estimator.waves_linearity_max)
        if len(linearity_coefficients[possible_values]) < 1:
            raise ValueError('No correct linearity coefficient found')
        index_linearity_criteria = np.argmin(linearity_coefficients[possible_values])
        travelled_distance = distances[possible_values][index_linearity_criteria]
        self.metrics['linearity_coefficients'] = linearity_coefficients
        self.metrics['celerities'] = celerities
        return travelled_distance

    def get_correlation_image(self) -> WavesImage:
        """ This function computes the correlation image by projecting the the correlation matrix
        on an array where axis are distances and center is the point where distance is 0.
        If several points have same coordinates, the mean of correlation is taken for this position
        """

        indices_x = np.round(self.distances * np.cos(np.radians(self.angles)))
        indices_x = np.array(indices_x - np.min(indices_x), dtype=int).T

        indices_y = np.round(self.distances * np.sin(np.radians(self.angles)))
        indices_y = np.array(indices_y - np.min(indices_y), dtype=int).T

        xr_s = pandas.Series(indices_x.flatten())
        yr_s = pandas.Series(indices_y.flatten())
        values_s = pandas.Series(self.correlation_matrix.flatten())

        # if two correlation values have same xr and yr mean of these values is taken
        dataframe = pandas.DataFrame({'xr': xr_s, 'yr': yr_s, 'values': values_s})
        dataframe_grouped = dataframe.groupby(by=['xr', 'yr']).mean().reset_index()
        values = np.array(dataframe_grouped['values'])
        indices_x = np.array(dataframe_grouped['xr'])
        indices_y = np.array(dataframe_grouped['yr'])

        projected_matrix = np.nanmean(self.correlation_matrix) * np.ones(
            (np.max(indices_x) + 1, np.max(indices_y) + 1))
        projected_matrix[indices_x, indices_y] = values
        return WavesImage(projected_matrix, self.spatial_resolution)

    @property
    def correlation_image(self) -> WavesImage:
        """ Correlation image
        :return: correlation image used to perform radon transformation
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

    def compute_wave_length(self, sinogram: np.ndarray) -> float:
        """ Wave length computation (in meter)
        :param sinogram : sinogram used to compute wave length
        :returns: wave length
        """
        min_wavelength = wavelength_offshore(self.global_estimator.waves_period_min, self.gravity)
        period, wave_length_zeros = find_period_from_zeros(
            sinogram, int(min_wavelength / self.spatial_resolution))
        wave_length = period * self.spatial_resolution

        if self.debug_sample:
            self.metrics['wave_length_zeros'] = wave_length_zeros
        return wave_length

    def compute_distances(self, sinogram: np.ndarray, wave_length: float,
                          nb_hops: int) -> np.ndarray:
        """ Propagated distance computation (in meter)
        Maxima are computed using peaks detection and the smallest nb_hops distances are selected
        :param sinogram: sinogram having maximum variance
        :param wave_length: wave_length computed on sinogram
        :param nb_hops : number of peaks to consider to compute distances
        :returns: np.ndarray of size nb_hops containing computed distances
        """
        x_axis = np.arange(-(len(sinogram) // 2), len(sinogram) // 2 + 1)
        interval = np.logical_and(x_axis * self.spatial_resolution > -wave_length,
                                  x_axis * self.spatial_resolution < wave_length)
        period = int(wave_length / self.spatial_resolution)
        max_sinogram = np.max(sinogram)
        tuning_parameters = self.local_estimator_params['TUNING']
        peaks, _ = find_peaks(sinogram, height=tuning_parameters['PEAK_DETECTION_HEIGHT_RATIO']
                              * max_sinogram,
                              distance=tuning_parameters['PEAK_DETECTION_DISTANCE_RATIO']
                              * period)
        distances = x_axis[peaks] * self.spatial_resolution
        index = np.argsort(np.abs(distances))
        if self.debug_sample:
            self.metrics['interval'] = interval
            self.metrics['x_axis'] = x_axis
            self.metrics['max_indices'] = peaks[index][:nb_hops]
        return distances[index][:nb_hops]
