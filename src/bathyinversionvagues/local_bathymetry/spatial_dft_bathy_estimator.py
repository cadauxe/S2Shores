# -*- coding: utf-8 -*-
""" Class managing the computation of waves fields from two images taken at a small time interval.

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 5 mars 2021
"""
from typing import Optional, List, Tuple, TYPE_CHECKING, cast  # @NoMove

from scipy.signal import find_peaks

import numpy as np

from ..bathy_debug.waves_fields_display import display_curve, display_4curves, display_3curves
from ..bathy_physics import wavenumber_offshore, phi_limits
from ..generic_utils.image_filters import detrend, desmooth
from ..image_processing.waves_image import WavesImage, ImageProcessingFilters
from ..image_processing.waves_radon import WavesRadon
from ..waves_exceptions import WavesEstimationError

from .local_bathy_estimator import LocalBathyEstimator
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

        self.directions_ranges = []

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

        peaks_dir_indices = self.find_directions()
        # self.find_directions_bis()

        self.prepare_refinement(peaks_dir_indices)

        self.find_spectral_peaks()

    def sort_waves_fields(self) -> None:
        """ Sort the waves fields estimations based on their energy max.
        """
        self.waves_fields_estimations.sort(key=lambda x: x.energy, reverse=True)

    def find_directions(self) -> np.ndarray:
        """ Find an initial set of directions from the cross correlation spectrum of the radon
        transforms of the 2 images.
        """

        # TODO: this processing sequence is related to bathymetry. Move elsewhere?
        kfft = self.radon_transforms[0].spectrum_wave_numbers
        phi_min, phi_max = self.get_phi_limits(kfft)

        # TODO: modify directions finding such that only one radon transform is computed (50% gain)
        sino1_fft = self.radon_transforms[0].get_sinograms_standard_dfts()
        sino2_fft = self.radon_transforms[1].get_sinograms_standard_dfts()
        _, total_spectrum, metrics = self._cross_correl_spectrum(sino1_fft, sino2_fft,
                                                                 phi_min, phi_max)
        total_spectrum_normalized = self.process_cross_correl_spectrum(total_spectrum, metrics)
        self.optimized_curve = total_spectrum_normalized
        # TODO: possibly apply symmetry to totalSpecMax_ref in find directions
        peaks, values = find_peaks(total_spectrum_normalized,
                                   prominence=self.local_estimator_params['PROMINENCE_MAX_PEAK'])
        prominences = values['prominences']

        # TODO: use symmetric peaks removal method (uncomment and delete next line.
#        peaks = self._process_peaks(peaks, prominences)
        if peaks.size == 0:
            raise WavesEstimationError('Unable to find any directional peak')

        if self.debug_sample:
            self.metrics['standard_dft'] = metrics

        return peaks

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

    def prepare_refinement(self, peaks_dir_indices: np.ndarray) -> None:
        """ Prepare the directions along which direction and wavenumber finding will be done.
        """
        if peaks_dir_indices.size > 0:
            for peak_index in range(0, peaks_dir_indices.size):
                angles_half_range = self.local_estimator_params['ANGLE_AROUND_PEAK_DIR']
                direction_index = peaks_dir_indices[peak_index]
                tmp = np.arange(max(direction_index - angles_half_range, 0),
                                min(direction_index + angles_half_range + 1, 360)
                                )
                directions_range = np.zeros_like(tmp)
                directions_range[:] = self.radon_transforms[0].directions[tmp]
                self.directions_ranges.append(directions_range)

        # FIXME: what to do with opposite directions

    def find_spectral_peaks(self) -> None:
        """ Find refined directions from the resampled cross correlation spectrum of the radon
        transforms of the 2 images and identify wavenumbers of the peaks along these directions.
        """
        # Detailed analysis of the signal for positive phase shifts

        kfft = self.get_kfft()
        phi_min, phi_max = self.get_phi_limits(kfft)

        for directions_range in self.directions_ranges:
            self._find_peaks_on_directions_range(kfft, phi_min, phi_max, directions_range)

        if self.debug_sample:
            self._metrics['kfft'] = kfft

    def _find_peaks_on_directions_range(self, kfft, phi_min, phi_max, directions) -> None:
        """ Find refined directions from the resampled cross correlation spectrum of the radon
        transforms of the 2 images and identify wavenumbers of the peaks along these directions.
        """
        # Detailed analysis of the signal for positive phase shifts
        self.radon_transforms[0].interpolate_sinograms_dfts(kfft, directions)
        self.radon_transforms[1].interpolate_sinograms_dfts(kfft, directions)
        sino1_fft = self.radon_transforms[0].get_sinograms_interpolated_dfts(directions)
        sino2_fft = self.radon_transforms[1].get_sinograms_interpolated_dfts(directions)
        phase_shift, total_spectrum, metrics = self._cross_correl_spectrum(sino1_fft, sino2_fft,
                                                                           phi_min, phi_max)
        total_spectrum_normalized = self.process_cross_correl_spectrum(total_spectrum, metrics)
        peaks_freq = find_peaks(total_spectrum_normalized,
                                prominence=self.local_estimator_params['PROMINENCE_MULTIPLE_PEAKS'])
        peaks_freq = peaks_freq[0]
        peaks_wavenumbers_ind = np.argmax(total_spectrum[:, peaks_freq], axis=0)

        for index, direction_index in enumerate(peaks_freq):
            wavenumber_index = peaks_wavenumbers_ind[index]
            estimated_phase_shift = phase_shift[wavenumber_index, direction_index]
            estimated_direction = \
                self.radon_transforms[0].directions[directions[direction_index]]
            peak_sinogram = self.radon_transforms[0][directions[direction_index]]

            # TODO: get wavelength from DFT frequencies
            normalized_frequency = peak_sinogram.interpolated_dft_frequencies[wavenumber_index]
            wavelength = 1 / (normalized_frequency * self.radon_transforms[0].sampling_frequency)
            waves_field_estimation = cast(SpatialDFTWavesFieldEstimation,
                                          self.create_waves_field_estimation(estimated_direction,
                                                                             wavelength))

            waves_field_estimation.delta_time = self.sequential_delta_times[0]
            waves_field_estimation.delta_phase = estimated_phase_shift
            waves_field_estimation.delta_phase_ratio = abs(waves_field_estimation.delta_phase) / \
                phi_max[wavenumber_index]

            waves_field_estimation.energy = total_spectrum[wavenumber_index, direction_index]
            self.store_estimation(waves_field_estimation)

        if self.debug_sample:
            self.metrics['kfft'] = kfft
            self.metrics['totSpec'] = np.abs(total_spectrum) / np.mean(total_spectrum)
            self.metrics['interpolated_dft'] = metrics

    def _cross_correl_spectrum(self, sino1_fft: np.ndarray, sino2_fft: np.ndarray,
                               phi_min: np.ndarray, phi_max: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """ Computes the cross correlation spectrum of the radon transforms of the images, possibly
        restricted to a limited set of directions.

        :param sino1_fft: the DFT of the first sinogram, either standard or interpolated
        :param sino2_fft: the DFT of the second sinogram, either standard or interpolated
        :param phi_min: minimum acceptable values of delta phi for each wavenumber to explore
        :param phi_max: maximum acceptable values of delta phi for each wavenumber to explore
        :returns: A tuple of 2 numpy arrays and a dictionary with:
                  - the phase shifts, thresholded by phi_min and phi_max
                  - the total spectrum
                  - a dictionary containing intermediate results for debugging purposes
        """
        nb_samples = sino1_fft.shape[0]

        sinograms_correlation_fft = sino2_fft * np.conj(sino1_fft)
        phase_shift = np.angle(sinograms_correlation_fft)

        phase_shift_thresholded = self.process_phase(phase_shift, phi_min, phi_max)

        amplitude_sino1 = np.abs(sino1_fft) ** 2
        amplitude_sino2 = np.abs(sino2_fft) ** 2
        combined_amplitude = (amplitude_sino1 + amplitude_sino2)

        # Find maximum total energy per direction theta and normalize by the greater one
        total_spectrum = np.abs(combined_amplitude * phase_shift_thresholded) / (nb_samples**3)
        metrics = {}
        metrics['sinograms_correlation_fft'] = sinograms_correlation_fft
        metrics['phase_shift_thresholded'] = phase_shift_thresholded
        metrics['total_spectrum'] = total_spectrum

        return phase_shift_thresholded, total_spectrum, metrics

    def process_cross_correl_spectrum(self, total_spectrum: np.ndarray, metrics: dict
                                      ) -> np.ndarray:
        """ Process the cross correlation spectrum of the radon transforms of the images,to
        derive the function used to locate maxima.

        :param total_spectrum: the cross correlation spectrum
        :returns:  the normalized spectrum
        """
        max_heta = np.max(total_spectrum, axis=0)
        total_spectrum_normalized = max_heta / np.max(max_heta)

        metrics['max_heta'] = max_heta
        metrics['total_spectrum_normalized'] = total_spectrum_normalized

        return total_spectrum_normalized

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
