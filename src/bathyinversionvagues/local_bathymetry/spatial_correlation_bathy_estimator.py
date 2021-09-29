# -*- coding: utf-8 -*-
""" Class performing bathymetry computation using spatial correlation method

:author: Degoul Romain
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 18/06/2021
"""
from typing import Optional, List, TYPE_CHECKING

from scipy.signal import find_peaks

import numpy as np

from ..bathy_physics import period_offshore
from ..generic_utils.image_filters import detrend, desmooth
from ..generic_utils.image_utils import normalized_cross_correlation
from ..generic_utils.signal_utils import find_period
from ..image_processing.waves_image import WavesImage, ImageProcessingFilters
from ..image_processing.waves_radon import WavesRadon
from ..image_processing.waves_sinogram import WavesSinogram

from .local_bathy_estimator import LocalBathyEstimator
from .spatial_correlation_waves_field_estimation import SpatialCorrelationWavesFieldEstimation
from .waves_fields_estimations import WavesFieldsEstimations


if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport


class SpatialCorrelationBathyEstimator(LocalBathyEstimator):
    """ Class performing spatial correlation to compute bathymetry
    """

    def __init__(self, images_sequence: List[WavesImage], global_estimator: 'BathyEstimator',
                 waves_fields_estimations: WavesFieldsEstimations,
                 selected_directions: Optional[np.ndarray] = None) -> None:

        super().__init__(images_sequence, global_estimator,
                         waves_fields_estimations, selected_directions)
        self._shape_x, self._shape_y = self.images_sequence[0].pixels.shape
        self._number_frames = len(self.images_sequence)

        self.radon_transforms: List[WavesRadon] = []
        self.sinograms: List[WavesSinogram] = []
        self.spatial_correlation = None
        self.directions = None

    def preprocessing_filters(self) -> ImageProcessingFilters:
        """
        """
        preprocessing_filters: ImageProcessingFilters = [(detrend, [])]

        if self.global_estimator.smoothing_requested:
            # FIXME: pixels necessary for smoothing are not taken into account, thus
            # zeros are introduced at the borders of the window.
            preprocessing_filters.append((desmooth,
                                          [self.global_estimator.smoothing_lines_size,
                                           self.global_estimator.smoothing_columns_size]))
            # Remove tendency possibly introduced by smoothing, specially on the shore line
            preprocessing_filters.append((detrend, []))
        return preprocessing_filters

    def create_waves_field_estimation(self, direction: float, wavelength: float,
                                      ) -> SpatialCorrelationWavesFieldEstimation:
        """ Creates the SpatialCorrelationWavesFieldEstimation instance where the local estimator
        will store its estimations.

        :param direction: the propagation direction of the waves field (degrees measured clockwise
                          from the North).
        :param wavelength: the wavelength of the waves field
        :returns: an initialized instance of WavesFilesEstimation to be filled in further on.
        """
        waves_field_estimation = SpatialCorrelationWavesFieldEstimation(
            self.gravity,
            self.global_estimator.depth_estimation_method,
            self.global_estimator.depth_estimation_precision)
        waves_field_estimation.direction = direction
        waves_field_estimation.wavelength = wavelength

        return waves_field_estimation

    def run(self) -> None:
        """
        """
        self.preprocess_images()  # TODO: should be in the init ?
        self.compute_radon_transforms()
        correlation_signal = self.compute_spatial_correlation()
        estimated_direction = self.find_direction()
        wavelength = self.compute_wavelength(estimated_direction)
        celerity = self.compute_celerity(wavelength)
        self.save_waves_field_estimation(estimated_direction, wavelength, celerity)

    def compute_radon_transforms(self) -> None:

        for image in self.images_sequence:
            radon_transform = WavesRadon(image)
            radon_transform.compute(self.selected_directions)
            radon_transform_augmented = \
                radon_transform.radon_augmentation(
                    self.local_estimator_params.AUGMENTED_RADON_FACTOR)
            self.radon_transforms.append(radon_transform_augmented)

    def find_direction(self) -> float:
        """
        """
        tmp_image = np.ones(self.images_sequence[0].pixels.shape)
        for frame in range(self._number_frames):
            tmp_image *= self.images_sequence[frame].pixels
        tmp_wavesimage = WavesImage(tmp_image, self.images_sequence[0].resolution)
        tmp_wavesradon = WavesRadon(tmp_wavesimage)
        tmp_wavesradon.compute()
        tmp_wavesradon_augmented = tmp_wavesradon.radon_augmentation(
            self.local_estimator_params.AUGMENTED_RADON_FACTOR)
        _, estimated_direction, _ = tmp_wavesradon_augmented.get_sinogram_maximum_variance()
        return estimated_direction

    def compute_spatial_correlation(self) -> np.ndarray:
        """
        """
        for radon_transform in self.radon_transforms:
            tmp_wavessinogram = radon_transform.get_sinogram(estimated_direction)
            tmp_wavessinogram.sinogram *= tmp_wavessinogram.variance
            self.sinograms.append(tmp_wavessinogram)
        sinogram_0 = self.sinograms[0]
        # TODO: should be independent from 0/1 (for multiple pairs of frames)
        sinogram_1 = self.sinograms[1]
        correl_mode = self.local_estimator_params.CORRELATION_MODE
        corr_init = normalized_cross_correlation(sinogram_0, sinogram_1, correl_mode)
        corr_init_ac = normalized_cross_correlation(corr_init, corr_init, correl_mode)
        corr_1 = normalized_cross_correlation(corr_init_ac, sinogram_1, correl_mode)
        corr_2 = normalized_cross_correlation(corr_init_ac, sinogram_2, correl_mode)
        correlation_signal = normalized_cross_correlation(corr_1, corr_2, correl_mode)
        return correlation_signal

    def compute_wavelength(self, correlation_signal: np.ndarray) -> float:
        """
        """
        period, _ = find_period(correlation_signal)  # Is it really a period ?
        wavelength = period * self.images_sequence[0].resolution * \
            self.local_estimator_params.AUGMENTED_RADON_FACTOR
        return wavelength

    def compute_celerity(self, wavelength: float) -> float:
        """
        """
        argmax_ac = len(corr)
        gover2pi = self.gravity / (2 * np.pi)
        peak_position_lim_inf = -gover2pi * \
            self.global_estimator.waves_period_max * abs(self.delta_time)
        if self.delta_time < period_offshore(1. / wavelength, self.gravity):
            peak_position_lim_sup = -peak_position_lim_inf
        else:
            # unused for s2
            propagation_factor = self.delta_time / period_offshore(1. / wavelength, self.gravity)
            peak_position_lim_sup = -self.local_estimator_params.PEAK_POSITION_MAX_FACTOR * \
                propagation_factor * wavelength
        # TODO: deal with  wavelength = 0 or np.nan
        peaks_pos, _ = find_peaks(corr)
        celerity = np.nan
        if peaks_pos.size != 0:
            relative_distance = peaks_pos - argmax_ac
            pt_in_range = peaks_pos[np.where((relative_distance >= peak_position_lim_inf) & (
                relative_distance < peak_position_lim_sup))]
            if pt_in_range.size != 0:
                argmax = pt_in_range[corr[pt_in_range].argmax()]
                dx = argmax - argmax_ac  # supposed to be in meters
                celerity = abs(dx) / abs(self.delta_time)
        return celerity

    def save_waves_field_estimation(self,
                                    direction: float,
                                    wavelength: float,
                                    celerity: float) -> None:
        """
        """
        # FIXME: create_waves_filed_estimation() could be replaced by a function
        # save_waves_field_estimation to avoid multiple saving functions (specially since we
        # have to create them anyway)
        waves_field_estimation = self.create_waves_field_estimation(direction, wavelength)
        waves_field_estimation.celerity = celerity
        # TODO: add more outputs (dx, corr)

        self.store_estimation(waves_field_estimation)
