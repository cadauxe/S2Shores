# -*- coding: utf-8 -*-
""" Class performing bathymetry computation using spatial correlation method

:author: Grégoire Thoumyre
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 20/09/2021
"""
from typing import Optional, List, TYPE_CHECKING, cast  # @NoMove

from scipy.signal import find_peaks

import numpy as np

from ..bathy_physics import celerity_offshore, wavelength_offshore, time_sampling_factor_offshore
from ..data_model.waves_fields_estimations import WavesFieldsEstimations
from ..generic_utils.image_filters import detrend, desmooth
from ..generic_utils.image_utils import normalized_cross_correlation
from ..generic_utils.signal_utils import find_period_from_zeros
from ..image_processing.sinograms import Sinograms
from ..image_processing.waves_image import WavesImage, ImageProcessingFilters
from ..image_processing.waves_radon import WavesRadon, linear_directions
from ..image_processing.waves_sinogram import WavesSinogram
from ..waves_exceptions import WavesEstimationError
from .local_bathy_estimator import LocalBathyEstimator
from .spatial_correlation_waves_field_estimation import SpatialCorrelationWavesFieldEstimation


if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport


class SpatialCorrelationBathyEstimator(LocalBathyEstimator):
    """ Class performing spatial correlation to compute bathymetry
    """

    waves_field_estimation_cls = SpatialCorrelationWavesFieldEstimation

    def __init__(self, images_sequence: List[WavesImage], global_estimator: 'BathyEstimator',
                 waves_fields_estimations: WavesFieldsEstimations,
                 selected_directions: Optional[np.ndarray] = None) -> None:

        super().__init__(images_sequence, global_estimator,
                         waves_fields_estimations, selected_directions)
        self._number_frames = len(self.images_sequence)
        if self.selected_directions is None:
            self.selected_directions = linear_directions(-180., 0., 1.)

        self.radon_transforms: List[Sinograms] = []
        self.sinograms: List[WavesSinogram] = []
        self.spatial_correlation = None
        self.directions = None

    @property
    def preprocessing_filters(self) -> ImageProcessingFilters:
        preprocessing_filters: ImageProcessingFilters = []
        preprocessing_filters.append((detrend, []))

        if self.global_estimator.smoothing_requested:
            # FIXME: pixels necessary for smoothing are not taken into account, thus
            # zeros are introduced at the borders of the window.
            preprocessing_filters.append((desmooth,
                                          [self.global_estimator.smoothing_lines_size,
                                           self.global_estimator.smoothing_columns_size]))
            # Remove tendency possibly introduced by smoothing, specially on the shore line
            preprocessing_filters.append((detrend, []))
        return preprocessing_filters

    def run(self) -> None:
        self.preprocess_images()  # TODO: should be in the init ?
        self.compute_radon_transforms()
        estimated_direction = self.find_direction()
        correlation_signal = self.compute_spatial_correlation(estimated_direction)
        wavelength = self.compute_wavelength(correlation_signal)
        propagated_distance = self.compute_propagated_distance(correlation_signal, wavelength)
        self.save_waves_field_estimation(estimated_direction, wavelength, propagated_distance)

    def compute_radon_transforms(self) -> None:

        for image in self.images_sequence:
            radon_transform = WavesRadon(image, self.selected_directions)
            radon_transform_augmented = radon_transform.radon_augmentation(
                self.local_estimator_params['AUGMENTED_RADON_FACTOR'])
            self.radon_transforms.append(radon_transform_augmented)

    def find_direction(self) -> float:
        """ Find the direction of the waves propagation

        :returns: the estimated direction of the waves propagation
        """
        tmp_image = np.ones(self.images_sequence[0].pixels.shape)
        for frame in range(self._number_frames):
            tmp_image *= self.images_sequence[frame].pixels
        tmp_wavesimage = WavesImage(tmp_image, self.spatial_resolution)
        tmp_wavesradon = WavesRadon(tmp_wavesimage, self.selected_directions)
        tmp_wavesradon_augmented = tmp_wavesradon.radon_augmentation(
            self.local_estimator_params['AUGMENTED_RADON_FACTOR'])
        estimated_direction, _ = tmp_wavesradon_augmented.get_direction_maximum_variance()
        return estimated_direction

    def compute_spatial_correlation(self, estimated_direction: float) -> np.ndarray:
        """ Compute the spatial cross correlation between the 2 sinograms of the estimated direction

        :param estimated_direction: the estimated direction of the waves propagation
        :returns: the correlation signal
        """
        for radon_transform in self.radon_transforms:
            tmp_wavessinogram = radon_transform[estimated_direction]
            tmp_wavessinogram.values *= tmp_wavessinogram.variance
            self.sinograms.append(tmp_wavessinogram)
        sinogram_1 = self.sinograms[0].values
        # TODO: should be independent from 0/1 (for multiple pairs of frames)
        sinogram_2 = self.sinograms[1].values
        correl_mode = self.local_estimator_params['CORRELATION_MODE']
        corr_init = normalized_cross_correlation(sinogram_1, sinogram_2, correl_mode)
        corr_init_ac = normalized_cross_correlation(corr_init, corr_init, correl_mode)
        corr_1 = normalized_cross_correlation(corr_init_ac, sinogram_1, correl_mode)
        corr_2 = normalized_cross_correlation(corr_init_ac, sinogram_2, correl_mode)
        correlation_signal = normalized_cross_correlation(corr_1, corr_2, correl_mode)
        return correlation_signal

    def compute_wavelength(self, correlation_signal: np.ndarray) -> float:
        """ Compute the wave length of the waves

        :param correlation_signal: spatial cross correlated signal
        :returns: the wave length (m)
        """
        min_wavelength = wavelength_offshore(self.global_estimator.waves_period_min, self.gravity)
        period, _ = find_period_from_zeros(correlation_signal, int(
            min_wavelength / self.spatial_resolution))
        wavelength = period * self.spatial_resolution * \
            self.local_estimator_params['AUGMENTED_RADON_FACTOR']
        return wavelength

    def compute_propagated_distance(self, correlation_signal: np.ndarray,
                                    wavelength: float) -> float:
        """ Compute the distance propagated over time by the waves

        :param correlation_signal: spatial cross correlated signal
        :param wavelength: the wave length (m)
        :returns: the distance propagated over time by the waves (m)
        """
        argmax_ac = len(correlation_signal) / 2
        delta_time = self.sequential_delta_times[0]
        celerity_offshore_max = celerity_offshore(self.global_estimator.waves_period_max,
                                                  self.gravity)
        # TODO: revisit signs management
        spatial_shift_offshore_min = -celerity_offshore_max * abs(delta_time)
        propagation_factor = time_sampling_factor_offshore(
            1. / wavelength, delta_time, self.gravity)
        if propagation_factor < 1:
            spatial_shift_offshore_max = -spatial_shift_offshore_min
        else:
            # unused for s2
            spatial_shift_offshore_max = -self.local_estimator_params['PEAK_POSITION_MAX_FACTOR'] \
                * propagation_factor * wavelength
        peaks_pos, _ = find_peaks(correlation_signal)
        propagated_distance = np.nan
        if peaks_pos.size != 0:
            relative_distance = peaks_pos - argmax_ac
            pt_in_range = peaks_pos[np.where((relative_distance >= spatial_shift_offshore_min) & (
                relative_distance < spatial_shift_offshore_max))]
            if pt_in_range.size != 0:
                argmax = pt_in_range[correlation_signal[pt_in_range].argmax()]
                # TODO: add variable to adapt to be in meters
                propagated_distance = argmax_ac - argmax  # supposed to be in meters,
            else:
                raise WavesEstimationError('Unable to find any directional peak')
        else:
            raise WavesEstimationError('Unable to find any directional peak')

        return propagated_distance

    def save_waves_field_estimation(self,
                                    estimated_direction: float,
                                    wavelength: float,
                                    propagated_distance: float) -> None:
        """ Saves the waves_field_estimation

        :param estimated_direction: the waves estimated propagation direction
        :param wavelength: the wave length of the waves
        :param propagated_distance: the distance propagated over time by the waves
        """
        waves_field_estimation = cast(SpatialCorrelationWavesFieldEstimation,
                                      self.create_waves_field_estimation(estimated_direction,
                                                                         wavelength))
        waves_field_estimation.delta_time = self.sequential_delta_times[0]
        waves_field_estimation.propagated_distance = propagated_distance
        self.waves_fields_estimations.append(waves_field_estimation)
