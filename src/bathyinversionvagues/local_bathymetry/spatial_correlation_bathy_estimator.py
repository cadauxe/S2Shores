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

from ..bathy_physics import wavenumber_offshore, phi_limits
from ..generic_utils.image_filters import detrend, desmooth
from ..generic_utils.numpy_utils import dump_numpy_variable
from ..image_processing.waves_image import WavesImage, ImageProcessingFilters
from ..image_processing.waves_radon import WavesRadon
from ..image_processing.waves_sinogram import WavesSinogram
from ..waves_exceptions import WavesEstimationError

from .local_bathy_estimator import LocalBathyEstimator, LocalBathyEstimatorDebug
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
        """ Constructor

        :param images_sequence: sequence of image used to compute bathymetry
        :param global_estimator: global estimator
        :param selected_directions: selected_directions: the set of directions onto which the
        sinogram must be computed
        """
        print('hey', flush=True)
        super().__init__(images_sequence, global_estimator, waves_fields_estimations, selected_directions)
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
        """ Creates the SpatialCorrelationWavesFieldEstimation instance where the local estimator will
        store its estimations.

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
        self.compute_spatial_correlation()
        estimated_direction = self.find_direction()
        wavelength = self.compute_wavelength()
        celerity = self.compute_celerity(wavelength)
        self.save_waves_field_estimation(estimated_direction, wavelength, celerity)


    def compute_radon_transforms(self) -> None:

        for image in self.images_sequence:
            radon_transform = WavesRadon(image)
            radon_transform.compute(self.selected_directions,
                                    augmented_radon_factor=self.local_estimator_params.AUGMENTED_RADON_FACTOR)
            self.radon_transforms.append(radon_transform)

    def normalized_cross_correlation(self, template, comparison) -> np.ndarray:
        """
        """
        comparison = (comparison - np.mean(comparison)) / np.std(comparison)
        template = (template - np.mean(template)) / np.std(template)

        norm_cross_corr = np.correlate(template, comparison,
                                       self.local_estimator_params.CORRELATION_MODE)
        size_sinogram = len(template)
        size_crosscorr = len(norm_cross_corr)
        indMin = (size_crosscorr - size_sinogram) // 2
        indMax = (size_crosscorr + size_sinogram) // 2
        norm_cross_corr = norm_cross_corr[indMin:indMax]/size_sinogram

        return norm_cross_corr

    def find_direction(self) -> float:
        """
        """
        tmp_image = np.ones(self.images_sequence[0].pixels.shape)
        for frame in range(self._number_frames):
            tmp_image *= self.images_sequence[frame].pixels
        tmp_wavesimage = WavesImage(tmp_image, self.images_sequence[0].resolution)
        tmp_wavesradon = WavesRadon(tmp_wavesimage)
        tmp_wavesradon.compute(augmented_radon_factor=self.local_estimator_params.AUGMENTED_RADON_FACTOR)
        _, estimated_direction, _ = tmp_wavesradon.get_sinogram_maximum_variance()
        for radon_transform in self.radon_transforms:
            tmp_wavessinogram = radon_transform.get_sinogram(estimated_direction)
            tmp_wavessinogram.sinogram *= tmp_wavessinogram.variance
            self.sinograms.append(tmp_wavessinogram)
        return estimated_direction

    def compute_spatial_correlation(self) -> np.ndarray:
        """
        """
        sinogram_0 = self.sinograms[0]
        sinogram_1 = self.sinograms[1] # TODO: should be independent from 0/1 (for multiple pairs of frames)
        corr_init    = self.normalized_cross_correlation(sinogram_0, sinogram_1)
        corr_init_ac = self.normalized_cross_correlation(corr_init, corr_init)
        corr_1       = self.normalized_cross_correlation(corr_init_ac, sinogram_1)
        corr_2       = self.normalized_cross_correlation(corr_init_ac, sinogram_2)
        self.spatial_correlation = self.normalized_cross_correlation(corr_1, corr_2)

    def compute_wavelength(self) -> float:
        """
        """
        distance_min = int((self.local_estimator_params.G/(2*np.pi)) *
                           self.global_estimator.waves_period_min ** 2)
        peaks_max, _ = find_peaks(np.abs(self.spatial_correlation),
                                  distance=distance_min)
        if peaks_max.size != 0:
            wavelength = 2 * np.mean(np.abs(np.diff(peaks_max)))
        else:
            wavelength = np.nan
        return wavelength

    def compute_celerity(self, wavelength: float) -> float:
        """
        """
        argmax_ac = len(corr)
        gover2pi = self.gravity / (2 * np.pi)
        lim_inf = -gover2pi * self.global_estimator.waves_period_max * abs(self.delta_time)
        if self.local_estimator_params.DT < (np.sqrt(wavelength / gover2pi)):
            lim_sup = -lim_inf
        else:
            factor = self.delta_time / (np.sqrt(wavelength / gover2pi))
            lim_sup = -self.local_estimator_params.PEAK_POSITION_MAX_FACTOR * factor * wavelength  # FIXME: 0.8 arbitrary taken cf GregS (move to config file)

        # TODO: deal with  wavelength = 0 or np.nan
        peaks_pos, _ = find_peaks(corr)
        celerity = np.nan
        if peaks_pos.size != 0:
            relative_distance = peaks_pos - argmax_ac
            pt_in_range = peaks_pos[np.where((relative_distance >= lim_inf) & (relative_distance < lim_sup))]
            if pt_in_range.size != 0:
                argmax = pt_in_range[corr[pt_in_range].argmax()]
                dx = argmax - argmax_ac
                celerity = abs(dx) / abs(self.delta_time)
        return celerity

    def save_waves_field_estimation(self,
                                    direction: float,
                                    wavelength: float,
                                    celerity: float):
        """
        """
        # FIXME: create_waves_filed_estimation() could be replaced by a function save_waves_field_estimation to avoid
        #  multiple saving functions (specially since we have to create them anyway)
        waves_field_estimation = self.create_waves_field_estimation(direction, wavelength)
        waves_field_estimation.celerity = celerity
        # TODO: add more outputs

        self.store_estimation(waves_field_estimation)
