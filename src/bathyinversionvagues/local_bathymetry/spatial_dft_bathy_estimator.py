# -*- coding: utf-8 -*-
""" Class managing the computation of waves fields from two images taken at a small time interval.

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 5 mars 2021
"""
import copy
from typing import Optional, List, Tuple, TYPE_CHECKING, cast  # @NoMove

from scipy.signal import find_peaks

import numpy as np

from ..bathy_debug.waves_fields_display import (display_curve, display_4curves,
                                                display_3curves, display_estimation,
                                                display_initial_data, display_radon_transforms)
from ..bathy_physics import wavenumber_offshore, phi_limits
from ..generic_utils.image_filters import detrend, desmooth
from ..generic_utils.numpy_utils import dump_numpy_variable
from ..image_processing.waves_image import WavesImage, ImageProcessingFilters
from ..image_processing.waves_radon import WavesRadon
from ..waves_exceptions import WavesEstimationError

from .local_bathy_estimator import LocalBathyEstimator, LocalBathyEstimatorDebug
from .spatial_dft_waves_field_estimation import SpatialDFTWavesFieldEstimation
from .waves_fields_estimations import WavesFieldsEstimations


if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport


class SpatialDFTBathyEstimator(LocalBathyEstimator):
    """ A local bathymetry estimator estimating bathymetry from the DFT of the sinograms in
    radon transforms.
    """

    waves_field_estimation_cls = SpatialDFTWavesFieldEstimation

    def __init__(self, images_sequence: List[WavesImage], global_estimator: 'BathyEstimator',
                 waves_fields_estimations: WavesFieldsEstimations,
                 selected_directions: Optional[np.ndarray] = None) -> None:

        super().__init__(images_sequence, global_estimator, waves_fields_estimations,
                         selected_directions)

        self.radon_transforms: List[WavesRadon] = []

        self.peaks_dir: Optional[np.ndarray] = None
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

    def compute_radon_transforms(self) -> None:

        for image in self.images_sequence:
            radon_transform = WavesRadon(image, self.selected_directions)
            self.radon_transforms.append(radon_transform)

    def run(self) -> None:
        """ Radon, FFT, find directional peaks, then do detailed DFT analysis to find
        detailed phase shifts per linear wave number (k*2pi)

        """
        self.preprocess_images()

        self.compute_radon_transforms()

        self.find_directions()
        # self.find_directions_bis()

        self.prepare_refinement()

        self.find_spectral_peaks()

    def sort_waves_fields(self) -> None:
        """ Sort the waves fields estimations based on their energy max.
        """
        self.waves_fields_estimations.sort(key=lambda x: x.energy, reverse=True)

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
        peaks, values = find_peaks(total_spectrum_normalized,
                                   prominence=self.local_estimator_params['PROMINENCE_MAX_PEAK'])
        prominences = values['prominences']

        # TODO: use symmetric peaks removal method (uncomment and delete next line.
#        self.peaks_dir = self._process_peaks(peaks, prominences)
        self.peaks_dir = peaks
        if self.peaks_dir.size == 0:
            raise WavesEstimationError('Unable to find any directional peak')
        if self.debug_sample:
            self._metrics['initial_sino1_fft'] = copy.deepcopy(self._metrics['sino1_fft'])
            self._metrics['initial_sino2_fft'] = copy.deepcopy(self._metrics['sino2_fft'])
            self._metrics['initial_phase_shift'] = copy.deepcopy(self._metrics['phase_shift'])
            self._metrics['initial_phase_shift_thresholded'] = \
                copy.deepcopy(self._metrics['phase_shift_thresholded'])
            self._metrics['initial_combined_amplitude'] = copy.deepcopy(
                self._metrics['combined_amplitude'])
            self._metrics['initial_total_spectrum_normalized'] = copy.deepcopy(
                self._metrics['total_spectrum_normalized'])
            self._metrics['initial_amplitude_sino1'] = copy.deepcopy(
                self._metrics['amplitude_sino1'])
            self._metrics['initial_total_spectrum'] = copy.deepcopy(self._metrics['total_spectrum'])

    def _process_peaks(self, peaks: np.ndarray, prominences: np.ndarray) -> np.ndarray:
        # Find pairs of symmetric directions
        if self.debug_sample:
            print('initial peaks: ', peaks)
        peaks_pairs = []
        for index1 in range(peaks.size - 1):
            for index2 in range(index1 + 1, peaks.size):
                if abs(peaks[index1] - peaks[index2]) == 180:
                    peaks_pairs.append((index1, index2))
                    break
        if self.debug_sample:
            print('peaks_pairs: ', peaks_pairs)

        filtered_peaks_dir = []
        # Keep only one direction from each pair, with the greatest prominence
        for index1, index2 in peaks_pairs:
            if abs(prominences[index1] - prominences[index2]) < 100:
                # Prominences almost the same, keep lowest index
                filtered_peaks_dir.append(peaks[index1])
            else:
                if prominences[index1] > prominences[index2]:
                    filtered_peaks_dir.append(peaks[index1])
                else:
                    filtered_peaks_dir.append(peaks[index2])
        if self.debug_sample:
            print('peaks kept from peaks_pairs: ', filtered_peaks_dir)

        # Add peaks which do not belong to a pair
        for index in range(peaks.size):
            found_in_pair = False
            for index1, index2 in peaks_pairs:
                if index == index1 or index == index2:
                    found_in_pair = True
                    break
            if not found_in_pair:
                filtered_peaks_dir.append(peaks[index])
        if self.debug_sample:
            print('final peaks after adding isolated peaks: ', sorted(filtered_peaks_dir))

        return np.array(sorted(filtered_peaks_dir))

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
        peaks_dir_indices = self.peaks_dir
        if peaks_dir_indices.size > 0:
            for peak_index in range(0, peaks_dir_indices.size):
                angles_half_range = self.local_estimator_params['ANGLE_AROUND_PEAK_DIR']
                direction_index = peaks_dir_indices[peak_index]
                tmp = np.arange(max(direction_index - angles_half_range, 0),
                                min(direction_index + angles_half_range + 1, 360)
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

        self.radon_transforms[0].compute_sinograms_dfts(self.directions, kfft)
        self.radon_transforms[1].compute_sinograms_dfts(self.directions, kfft)
        phase_shift, total_spectrum, total_spectrum_normalized = \
            self.normalized_cross_correl_spectrum(phi_min, phi_max, interpolated_dft=True)
        peaks_freq = find_peaks(total_spectrum_normalized,
                                prominence=self.local_estimator_params['PROMINENCE_MULTIPLE_PEAKS'])
        peaks_freq = peaks_freq[0]
        peaks_wavenumbers_ind = np.argmax(total_spectrum[:, peaks_freq], axis=0)

        for index, peak_freq_index in enumerate(peaks_freq):
            peak_wavenumber_index = peaks_wavenumbers_ind[index]
            estimated_phase_shift = phase_shift[peak_wavenumber_index, peak_freq_index]
            estimated_direction = \
                self.radon_transforms[0].directions[self.directions[peak_freq_index]]

            wavelength = 1 / kfft[peak_wavenumber_index]
            waves_field_estimation = cast(SpatialDFTWavesFieldEstimation,
                                          self.create_waves_field_estimation(estimated_direction,
                                                                             wavelength))

            waves_field_estimation.delta_time = self.sequential_delta_times[0]
            waves_field_estimation.delta_phase = estimated_phase_shift
            waves_field_estimation.delta_phase_ratio = abs(waves_field_estimation.delta_phase) / \
                phi_max[peak_wavenumber_index]

            waves_field_estimation.energy = total_spectrum[peak_wavenumber_index, peak_freq_index]
            self.store_estimation(waves_field_estimation)

        if self.debug_sample:
            self._metrics['kfft'] = kfft
            self._metrics['totSpec'] = np.abs(total_spectrum) / np.mean(total_spectrum)

    def normalized_cross_correl_spectrum(self, phi_min: np.ndarray, phi_max: np.ndarray,
                                         interpolated_dft: bool = False
                                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Computes the cross correlation spectrum of the radon transforms of the images, possibly
        restricted to a limited set of directions, and derive the function used to locate maxima.

        :param phi_min: minimum acceptable values of delta phi for each wavenumber to explore
        :param phi_max: maximum acceptable values of delta phi for each wavenumber to explore
        :param interpolated_dft: a flag allowing to select the standard DFT or the interpolated DFT
        :returns: A tuple of 3 numpy arrays with:
                  - the phase shifts, thresholded by phi_min and phi_max
                  - the total spectrum
                  - the normalized spectrum
        """
        sino1_fft = self.radon_transforms[0].get_sinograms_dfts(self.directions, interpolated_dft)
        sino2_fft = self.radon_transforms[1].get_sinograms_dfts(self.directions, interpolated_dft)
        nb_samples = sino1_fft.shape[0]

        sinograms_correlation_fft = sino2_fft * np.conj(sino1_fft)
        phase_shift = np.angle(sinograms_correlation_fft)

        phase_shift_thresholded = self.process_phase(phase_shift, phi_min, phi_max)

        amplitude_sino1 = np.abs(sino1_fft) ** 2
        amplitude_sino2 = np.abs(sino2_fft) ** 2
        combined_amplitude = (amplitude_sino1 + amplitude_sino2)

        # Find maximum total energy per direction theta and normalize by the greater one
        total_spectrum = np.abs(combined_amplitude * phase_shift_thresholded) / (nb_samples**3)
        max_heta = np.max(total_spectrum, axis=0)
        total_spectrum_normalized = max_heta / np.max(max_heta)
        # Pick the maxima

        if self.debug_sample:
            self._metrics['sino1_fft'] = sino1_fft
            self._metrics['sino2_fft'] = sino2_fft
            self._metrics['phase_shift'] = phase_shift
            self._metrics['phase_shift_thresholded'] = phase_shift_thresholded
            self._metrics['combined_amplitude'] = combined_amplitude
            self._metrics['total_spectrum_normalized'] = total_spectrum_normalized
            self._metrics['amplitude_sino1'] = amplitude_sino1
            self._metrics['total_spectrum'] = total_spectrum

        return phase_shift_thresholded, total_spectrum, total_spectrum_normalized

    def process_phase(self, phase_shift: np.ndarray, phi_min: np.ndarray, phi_max: np.ndarray
                      ) -> np.ndarray:
        """ Thresholding of the phase shifts, possibly with phase unwraping (not implemented)

        :param phase_shift: the phase shifts coming from the cross correlation spectrum
        :param phi_min: minimum acceptable values of delta phi for each wavenumber
        :param phi_max: maximum acceptable values of delta phi for each wavenumber
        :returns: the thresholded phase shifts
        """

        if not self.local_estimator_params['UNWRAP_PHASE_SHIFT']:
            # currently deactivated but we want this functionality:
            result = np.copy(phase_shift)
        else:
            result = (phase_shift + 2 * np.pi) % (2 * np.pi)

        nb_directions = phase_shift.shape[1]

        # Deep water limitation [if the wave travels faster than the deep-water
        # limit we consider it non-physical]
        phi_max = np.tile(phi_max[:, np.newaxis], (1, nb_directions))
        # Minimal propagation speed; this depends on the Satellite; Venus or Sentinel 2
        phi_min = np.tile(phi_min[:, np.newaxis], (1, nb_directions))

        phase_shift_valid = (((phi_min < phase_shift) & (phase_shift < phi_max)) |
                             ((phi_max < phase_shift) & (phase_shift < phi_min)))
        result[np.logical_not(phase_shift_valid)] = 0

        return result

    def get_kfft(self) -> np.ndarray:
        """  :returns: the requested sampling of the sinogram FFT
        """
        # frequencies based on wave characteristics:
        period_samples = np.arange(self.global_estimator.waves_period_min,
                                   self.global_estimator.waves_period_max,
                                   self.local_estimator_params['STEP_T'])
        k_forced = cast(np.ndarray, wavenumber_offshore(period_samples, self.gravity))

        return k_forced

    def get_phi_limits(self, wavenumbers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """  Get the delta phase limits form deep and swallow waters

        :param wavenumbers: the wavenumbers for which limits on phase are requested
        :returns: the minimum and maximum phase shifts for swallow and deep water at different
                  wavenumbers
        """
        return cast(Tuple[np.ndarray, np.ndarray],
                    phi_limits(wavenumbers,
                               self.sequential_delta_times[0],
                               self.global_estimator.depth_min,
                               self.gravity))


class SpatialDFTBathyEstimatorDebug(LocalBathyEstimatorDebug, SpatialDFTBathyEstimator):
    """ Class allowing to debug the estimations made by a SpatialDFTBathyEstimator
    """

    def explore_results(self) -> None:
        self._dump_cross_correl_spectrum()

    def _dump_cross_correl_spectrum(self) -> None:
        metrics = self.metrics
        sino1_fft = metrics['sino1_fft']
        sino2_fft = metrics['sino2_fft']

        initial_sino1_fft = metrics['initial_sino1_fft']
        initial_sino2_fft = metrics['initial_sino2_fft']
        initial_total_spectrum_normalized = metrics['initial_total_spectrum_normalized']
        initial_phase_shift = metrics['initial_phase_shift']

        phase_shift = metrics['phase_shift']
        phase_shift_thresholded = metrics['phase_shift_thresholded']
        combined_amplitude = metrics['combined_amplitude']
        total_spectrum_normalized = metrics['total_spectrum_normalized']
        amplitude_sino1 = metrics['amplitude_sino1']
        total_spectrum = metrics['total_spectrum']

        # Printouts
        dump_numpy_variable(self.radon_transforms[0].pixels, 'input pixels for Radon transform 1 ')
        radon_array, directions = self.radon_transforms[0].get_as_arrays()
        dump_numpy_variable(radon_array, 'Radon transform 1')
        dump_numpy_variable(directions, 'Directions used for Radon transform 1')

        dump_numpy_variable(initial_sino1_fft, 'Initial sinoFFT1')
        dump_numpy_variable(initial_total_spectrum_normalized, 'initial_total_spectrum_normalized')
        dump_numpy_variable(initial_phase_shift, 'initial_phase_shift')

        dump_numpy_variable(sino1_fft, 'refined sinoFFT1')
        dump_numpy_variable(phase_shift, 'refined phase shift')
        for index in range(0, phase_shift.shape[1]):
            print(phase_shift[0][index])

        dump_numpy_variable(phase_shift_thresholded, 'refined phase shift thresholded')
        for index in range(0, phase_shift_thresholded.shape[1]):
            print(index, phase_shift_thresholded[1][index])

        dump_numpy_variable(combined_amplitude, 'refined combined_amplitude')
        dump_numpy_variable(total_spectrum_normalized, 'refined total_spectrum_normalized')

        # Displays
        display_initial_data(self)

        initial_sino1_directions = self.radon_transforms[0].directions
        initial_sino2_directions = self.radon_transforms[1].directions
        display_radon_transforms(self, initial_sino1_fft, self.directions,
                                 initial_sino2_fft, self.directions)
        display_curve(initial_total_spectrum_normalized, 'initial_total_spectrum_normalized')
        display_radon_transforms(self, sino1_fft, self.directions,
                                 sino2_fft, self.directions)
        display_curve(total_spectrum_normalized, 'total_spectrum_normalized')
        display_estimation(combined_amplitude, amplitude_sino1,
                           phase_shift,
                           phase_shift_thresholded, total_spectrum,
                           total_spectrum_normalized)

        if self.peaks_dir is not None:
            dump_numpy_variable(self.peaks_dir, 'found directions')
        else:
            print('No directions found !!!')

        print(f'estimations after direction refinement :')
        print(self.waves_fields_estimations)
