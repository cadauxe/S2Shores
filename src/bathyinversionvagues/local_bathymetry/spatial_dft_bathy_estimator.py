# -*- coding: utf-8 -*-
""" Class managing the computation of waves fields from two images taken at a small time interval.

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 5 mars 2021
"""
from typing import Optional, List, Tuple, TYPE_CHECKING  # @NoMove

from scipy.signal import find_peaks

import numpy as np

from ..bathy_physics import wavenumber_offshore, phi_limits
from ..generic_utils.image_filters import detrend, desmooth
from ..generic_utils.numpy_utils import dump_numpy_variable
from ..image_processing.waves_image import WavesImage, ImageProcessingFilters
from ..image_processing.waves_radon import WavesRadon
from ..waves_exceptions import WavesEstimationError
from ..waves_fields_display import (display_curve, display_4curves,
                                    display_3curves, display_estimation)

from .local_bathy_estimator import LocalBathyEstimator
from .spatial_dft_waves_field_estimation import SpatialDFTWavesFieldEstimation
from .waves_fields_estimations import WavesFieldsEstimations


if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport


class SpatialDFTBathyEstimator(LocalBathyEstimator):
    """ A local bathymetry estimator estimating bathymetry from the DFT of the sinograms in
    radon transforms.
    """

    def __init__(self, images_sequence: List[WavesImage], global_estimator: 'BathyEstimator',
                 waves_fields_estimations: WavesFieldsEstimations,
                 selected_directions: Optional[np.ndarray] = None) -> None:

        super().__init__(images_sequence, global_estimator, waves_fields_estimations,
                         selected_directions)

        self.radon_transforms: List[WavesRadon] = []

        self.peaks_dir = None
        self.directions = None

    def create_waves_field_estimation(self, direction: float, wavelength: float
                                      ) -> SpatialDFTWavesFieldEstimation:
        """ Creates the WavesFieldEstimation instance where the local estimator will store its
        estimations.

        :param direction: the propagation direction of the waves field (degrees measured clockwise
                          from the North).
        :param wavelength: the wavelength of the waves field
        :returns: an initialized instance of WavesFilesEstimation to be filled in further on.
        """
        waves_field_estimation = SpatialDFTWavesFieldEstimation(self.gravity,
                                                                self.local_estimator_params.DEPTH_EST_METHOD,
                                                                self.local_estimator_params.D_PRECISION)
        waves_field_estimation.direction = direction
        waves_field_estimation.wavelength = wavelength

        return waves_field_estimation

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

    def compute_radon_transforms(self) -> None:

        for image in self.images_sequence:
            radon_transform = WavesRadon(image)
            radon_transform.compute(self.selected_directions)
            self.radon_transforms.append(radon_transform)

    def run(self) -> None:
        """    Radon, FFT, find directional peaks, then do detailed DFT analysis to find
        detailed phase shifts per linear wave number (k *2pi)

        """
        self.preprocess_images()

        self.compute_radon_transforms()

        self.find_directions()
        # self.find_directions_bis()

        self.prepare_refinement()

        self.find_spectral_peaks()

        self._metrics['N'] = self.radon_transforms[0].nb_samples
        self._metrics['radon_image1'] = self.radon_transforms[0]
        self._metrics['radon_image2'] = self.radon_transforms[1]

    def sort_waves_fields(self) -> None:
        """ Sort the waves fields estimations based on their energy max.
        """
        self.waves_fields_estimations.sort(key=lambda x: x.energy_max, reverse=True)

    def find_directions(self) -> None:
        """ Find an initial set of directions from the cross correlation spectrum of the radon
        transforms of the 2 images.
        """

        # TODO: this processing sequence is related to bathymetry. Move elsewhere?
        kfft = self.radon_transforms[0].spectrum_wave_numbers
        phi_min, phi_max = self.get_phi_limits(kfft)

        # TODO: modify directions finding such that only one radon transform is computed (50% gain)
        self.radon_transforms[0].compute_sinograms_dfts()
        self.radon_transforms[1].compute_sinograms_dfts()
        _, _, total_spectrum_normalized = self.normalized_cross_correl_spectrum(phi_min, phi_max)
        self.optimized_curve = total_spectrum_normalized
        # TODO: possibly apply symmetry to totalSpecMax_ref in find directions
        self.peaks_dir = find_peaks(total_spectrum_normalized,
                                    prominence=self.local_estimator_params.PROMINENCE_MAX_PEAK)
        if len(self.peaks_dir[0]) == 0:  # pylint: disable=len-as-condition
            raise WavesEstimationError('Unable to find any directional peak')
        if self.debug_sample:
            print('initial directions')
            print(self.peaks_dir)

    def find_directions_bis(self) -> None:
        """ Find an initial set of directions from the the radon transform of a single image.
        Exploratory test.
        """

        sinograms_powers_normalized_list: List[np.ndarray] = []
        for radon_transform in self.radon_transforms:
            sinograms_powers = radon_transform.get_sinograms_mean_power()
            sinograms_powers_normalized = sinograms_powers / np.max(sinograms_powers)
            sinograms_powers_normalized_list.append(sinograms_powers_normalized)
            peaks_direction_radon = find_peaks(sinograms_powers_normalized, prominence=0.1)
            print(peaks_direction_radon)

        for index, radon_transform in enumerate(self.radon_transforms):
            display_curve(sinograms_powers_normalized_list[index],
                          f'Power sinograms image {index+1}')
        display_3curves(
            self.optimized_curve,
            sinograms_powers_normalized_list[0],
            sinograms_powers_normalized_list[1])
        derived_metric = sinograms_powers_normalized_list[0] * sinograms_powers_normalized_list[1]
        peaks_derived_metric = find_peaks(derived_metric, prominence=0.05)
        print(peaks_derived_metric)
        display_4curves(self.optimized_curve,
                        derived_metric,
                        sinograms_powers_normalized[0][::5],
                        sinograms_powers_normalized[1][::5])

    def prepare_refinement(self) -> None:
        """ Prepare the directions along which direction and wavenumber finding will be done.
        """
        refined_directions: List[np.ndarray] = []
        peaks_dir_indices = self.peaks_dir[0]
        if peaks_dir_indices.size > 0:
            for peak_index in range(0, peaks_dir_indices.size):
                angles_half_range = self.local_estimator_params.ANGLE_AROUND_PEAK_DIR
                tmp = np.arange(np.max([peaks_dir_indices[peak_index] - angles_half_range, 0]),
                                np.min([peaks_dir_indices[peak_index] + angles_half_range + 1, 180])
                                )
                if peak_index == 0:
                    refined_directions = tmp
                else:
                    refined_directions = np.append(refined_directions, tmp)
        # delete double directions
        # FIXME: this reorders the directions which is not always desired
        directions_indices = np.unique(refined_directions)
        self.directions = np.zeros_like(directions_indices)
        self.directions[:] = self.radon_transforms[0].directions[directions_indices]

    def find_spectral_peaks(self) -> None:
        """ Find refined directions from the resampled cross correlation spectrum of the radon
        transforms of the 2 images and identify wavenumbers of the peaks along these directions.
        """
        # Detailed analysis of the signal for positive phase shifts

        kfft = self.get_kfft()
        phi_min, phi_max = self.get_phi_limits(kfft)
        self._metrics['kfft'] = kfft

        self.radon_transforms[0].compute_sinograms_dfts(self.directions, kfft)
        self.radon_transforms[1].compute_sinograms_dfts(self.directions, kfft)
        phase_shift, total_spectrum, total_spectrum_normalized = \
            self.normalized_cross_correl_spectrum(phi_min, phi_max)
        peaks_freq = find_peaks(total_spectrum_normalized,
                                prominence=self.local_estimator_params.PROMINENCE_MULTIPLE_PEAKS)
        peaks_freq = peaks_freq[0]
        peaks_wavenumbers_ind = np.argmax(total_spectrum[:, peaks_freq], axis=0)
        self._metrics['totSpec'] = np.abs(total_spectrum) / np.mean(total_spectrum)

        for index, peak_freq_index in enumerate(peaks_freq):
            peak_wavenumber_index = peaks_wavenumbers_ind[index]
            estimated_phase_shift = phase_shift[peak_wavenumber_index, peak_freq_index]
            estimated_direction = \
                self.radon_transforms[0].directions[self.directions[peak_freq_index]]

            wavelength = 1 / kfft[peak_wavenumber_index][0]
            waves_field_estimation = self.create_waves_field_estimation(estimated_direction,
                                                                        wavelength)

            waves_field_estimation.delta_time = self.delta_time
            waves_field_estimation.delta_phase = estimated_phase_shift
            waves_field_estimation.delta_phase_ratio = abs(waves_field_estimation.delta_phase) / \
                phi_max[peak_wavenumber_index]
            waves_field_estimation.energy_max = total_spectrum_normalized[peak_freq_index]
            self.store_estimation(waves_field_estimation)
        self.print_estimations_debug('after direction refinement')

    def normalized_cross_correl_spectrum(self, phi_min: np.ndarray, phi_max: np.ndarray
                                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Computes the cross correlation spectrum of the radon transforms of the images, possibly
        restricted to a limited set of directions, and derive the function used to locate maxima.

        :param phi_min: minimum acceptable values of delta phi for each wavenumber to explore
        :param phi_max: maximum acceptable values of delta phi for each wavenumber to explore
        :returns: A tuple of 3 numpy arrays with:
                  - the phase shifts, thresholded by phi_min and phi_max
                  - the total spectrum
                  - the normalized spectrum
        """

        if self.directions is None:
            nb_directions = self.radon_transforms[0].nb_directions
        else:
            nb_directions = self.directions.shape[0]
        phi_min = np.tile(phi_min[:, np.newaxis], (1, nb_directions))
        phi_max = np.tile(phi_max[:, np.newaxis], (1, nb_directions))

        sino1_fft = self.radon_transforms[0].get_sinograms_dfts(self.directions)
        sino2_fft = self.radon_transforms[1].get_sinograms_dfts(self.directions)

        sinograms_correlation_fft = sino2_fft * np.conj(sino1_fft)
        phase_shift = np.angle(sinograms_correlation_fft)
        phase_shift_thresholded = self.process_phase(phase_shift, phi_min, phi_max)

        amplitude_sino1 = np.abs(sino1_fft) ** 2
        amplitude_sino2 = np.abs(sino2_fft) ** 2
        combined_amplitude = (amplitude_sino1 + amplitude_sino2)

        # Find maximum total energy per direction theta and normalize by the greater one
        total_spectrum = np.abs(combined_amplitude * phase_shift_thresholded)
        max_heta = np.max(total_spectrum, axis=0)
        total_spectrum_normalized = max_heta / np.max(max_heta)
        # Pick the maxima

        if self.debug_sample:
            self._dump_cross_correl_spectrum(sino1_fft, phase_shift, phase_shift_thresholded,
                                             combined_amplitude, total_spectrum_normalized,
                                             amplitude_sino1, total_spectrum)

        return phase_shift_thresholded, total_spectrum, total_spectrum_normalized

    def process_phase(self, phase_shift: np.ndarray, phi_min: np.ndarray, phi_max: np.ndarray
                      ) -> np.ndarray:
        """ Thresholding of the phase shifts, possibly with phase unwraping (not implemented)

        :param phase_shift: the phase shifts coming from the cross correlation spectrum
        :param phi_min: minimum acceptable values of delta phi for each wavenumber
        :param phi_max: maximum acceptable values of delta phi for each wavenumber
        :returns: the thresholded phase shifts
        """

        if not self.local_estimator_params.UNWRAP_PHASE_SHIFT:
            # currently deactivated but we want this functionality:
            # FIXME: should we take absolute value here?
            result = np.copy(phase_shift)
        else:
            result = (phase_shift + 2 * np.pi) % (2 * np.pi)
        # Deep water limitation [if the wave travels faster that the deep-water
        # limit we consider it non-physical]
        result[np.abs(phase_shift) > phi_max] = 0
        # Minimal propagation speed; this depends on the Satellite; Venus or Sentinel 2
        result[np.abs(phase_shift) < phi_min] = 0
        return result

    def get_kfft(self) -> np.ndarray:
        """  :returns: the requested sampling of the sinogram FFT
        """
        # frequencies based on wave characteristics:
        period_samples = np.arange(self.local_estimator_params.MIN_T,
                                   self.local_estimator_params.MAX_T,
                                   self.local_estimator_params.STEP_T)
        k_forced = wavenumber_offshore(period_samples, self.gravity)

        return k_forced.reshape((k_forced.size, 1))

    def get_phi_limits(self, wavenumbers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """  Get the delta phase limits form deep and swallow waters

        :param wavenumbers: the wavenumbers for which limits on phase are requested
        :returns: the minimum and maximum phase shifts for swallow and deep water at different
                  wavenumbers
        """
        return phi_limits(wavenumbers,
                          self.delta_time,
                          self.local_estimator_params.MIN_D,
                          self.gravity)

    def _dump_cross_correl_spectrum(self, sino1_fft, phase_shift, phase_shift_thresholded,
                                    combined_amplitude, total_spectrum_normalized, amplitude_sino1,
                                    total_spectrum) -> None:
        dump_numpy_variable(self.radon_transforms[0].pixels, 'Radon transform 1 input pixels')
        dump_numpy_variable(
            self.radon_transforms[0].radon_transform.get_as_array(),
            'Radon transform 1')
        dump_numpy_variable(self.directions, 'Directions used for radon transform 1')
        dump_numpy_variable(sino1_fft, 'sinoFFT1')
        dump_numpy_variable(phase_shift, 'phase shift')
        for index in range(0, phase_shift.shape[1]):
            print(phase_shift[0][index])

        dump_numpy_variable(phase_shift_thresholded, 'phase shift thresholded')
        for index in range(0, phase_shift_thresholded.shape[1]):
            print(index, phase_shift_thresholded[1][index])

        dump_numpy_variable(combined_amplitude, 'combined_amplitude')
        dump_numpy_variable(total_spectrum_normalized, 'total_spectrum_normalized')

        display_curve(total_spectrum_normalized, 'total_spectrum_normalized')
        display_estimation(
            combined_amplitude, amplitude_sino1,
            phase_shift,
            phase_shift_thresholded, total_spectrum,
            total_spectrum_normalized)
