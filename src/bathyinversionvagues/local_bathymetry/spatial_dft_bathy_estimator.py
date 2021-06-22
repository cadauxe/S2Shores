# -*- coding: utf-8 -*-
""" Class managing the computation of waves fields from two images taken at a small time interval.

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 5 mars 2021
"""
from typing import Optional, List  # @NoMove

from scipy.signal import find_peaks

import numpy as np

from ..image_processing.waves_image import WavesImage
from ..image_processing.waves_radon import WavesRadon
from ..waves_exceptions import WavesEstimationError
from ..waves_fields_display import (display_curve, display_4curves,
                                    display_3curves, display_estimation)

from .local_bathy_estimator import LocalBathyEstimator
from .waves_field_estimation import WavesFieldEstimation


class SpatialDFTBathyEstimator(LocalBathyEstimator):
    # TODO: change detrend by passing a pre_processing function, with optional parameters
    def __init__(self, images_sequence: List[WavesImage], global_estimator,
                 selected_directions: Optional[np.ndarray] = None) -> None:
        """ Constructor

        :param selected_directions: the set of directions onto which the sinogram must be computed
        """
        super().__init__(images_sequence, global_estimator, selected_directions)

        self.radon_transforms: List[WavesRadon] = []
        for image in images_sequence:
            radon_transform = WavesRadon(image)
            radon_transform.compute(selected_directions)
            self.radon_transforms.append(radon_transform)

        # delta time between the two images in seconds
        self.delta_time = self.global_estimator.waveparams.DT

        self.peaks_dir = None
        self.directions = None

    def run(self) -> None:
        """    Radon, FFT, find directional peaks, then do detailed DFT analysis to find
        detailed phase shifts per linear wave number (k *2pi)

        """
        self.find_directions()

#            waves_fields_estimator.find_directions_bis()

        self.prepare_refinement()

        self.find_spectral_peaks()

        self._metrics['N'] = self.radon_transforms[0].nb_samples
        self._metrics['radon_image1'] = self.radon_transforms[0]
        self._metrics['radon_image2'] = self.radon_transforms[1]

    def find_directions(self) -> None:

        # TODO: this processing sequence is related to bathymetry. Move elsewhere.
        phi_max, phi_min = self.global_estimator.get_phi_limits(
            self.radon_transforms[0].spectrum_wave_numbers)

        # TODO: modify directions finding such that only one radon transform is computed (50% gain)
        self.radon_transforms[0].compute_sinograms_dfts()
        self.radon_transforms[1].compute_sinograms_dfts()
        _, _, totalSpecMax_ref = self.normalized_cross_correl_spectrum(phi_min, phi_max)
        self.optimized_curve = totalSpecMax_ref
        # TODO: possibly apply symmetry to totalSpecMax_ref in find directions
        self.peaks_dir = find_peaks(totalSpecMax_ref,
                                    prominence=self.local_estimator_params.PROMINENCE_MAX_PEAK)
        if len(self.peaks_dir[0]) == 0:  # pylint: disable=len-as-condition
            raise WavesEstimationError('Unable to find any directional peak')
        if self.global_estimator.debug_sample:
            print('initial directions')
            print(self.peaks_dir)

    def find_directions_bis(self) -> None:

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
        refined_directions: List[np.ndarray] = []
        peaks_dir_indices = self.peaks_dir[0]
        if peaks_dir_indices.size > 0:
            for ii in range(0, peaks_dir_indices.size):
                angles_half_range = self.local_estimator_params.ANGLE_AROUND_PEAK_DIR
                tmp = np.arange(np.max([peaks_dir_indices[ii] - angles_half_range, 0]),
                                np.min([peaks_dir_indices[ii] + angles_half_range + 1, 180]))
                if ii == 0:
                    refined_directions = tmp
                else:
                    refined_directions = np.append(refined_directions, tmp)
        # delete double directions
        # FIXME: this reorders the directions which is not always desired
        directions_indices = np.unique(refined_directions)
        self.directions = np.zeros_like(directions_indices)
        self.directions[:] = self.radon_transforms[0].directions[directions_indices]

    def find_spectral_peaks(self) -> None:
                # Detailed analysis of the signal for positive phase shifts

        kfft = self.global_estimator.get_kfft()
        phi_max, phi_min = self.global_estimator.get_phi_limits()
        self._metrics['kfft'] = kfft

        self.radon_transforms[0].compute_sinograms_dfts(self.directions, kfft)
        self.radon_transforms[1].compute_sinograms_dfts(self.directions, kfft)
        phase_shift, totSpec, totalSpecMax_ref = self.normalized_cross_correl_spectrum(phi_min,
                                                                                       phi_max)
        peaksFreq = find_peaks(totalSpecMax_ref,
                               prominence=self.local_estimator_params.PROMINENCE_MULTIPLE_PEAKS)
        peaksFreq = peaksFreq[0]
        peaksK = np.argmax(totSpec[:, peaksFreq], axis=0)
        self._metrics['totSpec'] = np.abs(totSpec) / np.mean(totSpec)

        for ii, peak_freq_index in enumerate(peaksFreq):
            waves_field_estimation = WavesFieldEstimation(
                self.delta_time,
                self.local_estimator_params.D_PRECISION,
                self.local_estimator_params.G,
                self.local_estimator_params.DEPTH_EST_METHOD)

            peak_wavenumber_index = peaksK[ii]
            estimated_phase_shift = phase_shift[peak_wavenumber_index, peak_freq_index]
            estimated_direction = \
                self.radon_transforms[0].directions[self.directions[peak_freq_index]]
            if estimated_phase_shift < 0.:
                estimated_direction += 180

            waves_field_estimation.direction = estimated_direction
            waves_field_estimation.delta_phase = abs(estimated_phase_shift)
            waves_field_estimation.delta_phase_ratio = waves_field_estimation.delta_phase / \
                phi_max[peak_wavenumber_index]
            waves_field_estimation.energy_max = totalSpecMax_ref[peak_freq_index]
            waves_field_estimation.wavenumber = kfft[peak_wavenumber_index][0]
            self.store_estimation(waves_field_estimation)
        self.print_estimations_debug('after direction refinement')

        # sort the waves fields by energy_max level
        self.waves_fields_estimations.sort(key=lambda x: x.energy_max, reverse=True)
        self.print_estimations_debug('after estimations sorting')

    def normalized_cross_correl_spectrum(self, phi_min, phi_max):

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
        totSpec = np.abs(combined_amplitude * phase_shift_thresholded)
        maxHeta = np.max(totSpec, axis=0)
        totalSpecMax_ref = maxHeta / np.max(maxHeta)
        # Pick the maxima

        if self.global_estimator.debug_sample:
            self._dump(self.radon_transforms[0].pixels, 'Radon transform 1 input pixels')
            self._dump(self.radon_transforms[0].radon_transform.get_as_array(), 'Radon transform 1')
            self._dump(self.directions, 'Directions used for radon transform 1')
            self._dump(sino1_fft, 'sinoFFT1')
            self._dump(phase_shift, 'phase shift')
            for index in range(0, phase_shift.shape[1]):
                print(phase_shift[0][index])

            self._dump(phase_shift_thresholded, 'phase shift thresholded')
            for index in range(0, phase_shift_thresholded.shape[1]):
                print(index, phase_shift_thresholded[1][index])

            self._dump(combined_amplitude, 'combined_amplitude')
            self._dump(totalSpecMax_ref, 'totalSpecMax_ref')

            display_curve(totalSpecMax_ref, 'Total Spec Max ref')
            display_estimation(
                combined_amplitude, amplitude_sino1,
                phase_shift,
                phase_shift_thresholded, totSpec,
                totalSpecMax_ref)

        return phase_shift_thresholded, totSpec, totalSpecMax_ref

    def process_phase(self, phase_shift, phi_min, phi_max):

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
