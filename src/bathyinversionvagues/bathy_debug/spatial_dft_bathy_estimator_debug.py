# -*- coding: utf-8 -*-
""" Class for debugging the Spatial DFT estimator.

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 5 mars 2021
"""
import numpy as np

from ..generic_utils.numpy_utils import dump_numpy_variable
from ..local_bathymetry.local_bathy_estimator import LocalBathyEstimatorDebug
from ..local_bathymetry.spatial_dft_bathy_estimator import SpatialDFTBathyEstimator
from .waves_fields_display import display_initial_data, display_radon_transforms, display_context


class SpatialDFTBathyEstimatorDebug(LocalBathyEstimatorDebug, SpatialDFTBathyEstimator):
    """ Class allowing to debug the estimations made by a SpatialDFTBathyEstimator
    """

    def explore_results(self) -> None:
        metrics = self.metrics

        initial_sino1_fft = self.radon_transforms[0].get_sinograms_dfts()
        initial_sino2_fft = self.radon_transforms[1].get_sinograms_dfts()
        initial_total_spectrum_normalized = metrics['standard_dft']['total_spectrum_normalized']
        initial_phase_shift = np.angle(metrics['standard_dft']['sinograms_correlation_fft'])

        sino1_fft = self.radon_transforms[0].get_sinograms_dfts(interpolated_dft=True)
        sino2_fft = self.radon_transforms[1].get_sinograms_dfts(interpolated_dft=True)
        phase_shift = np.angle(metrics['interpolated_dft']['sinograms_correlation_fft'])
        phase_shift_thresholded = metrics['interpolated_dft']['phase_shift_thresholded']
        combined_amplitude = metrics['interpolated_dft']['combined_amplitude']
        total_spectrum_normalized = metrics['interpolated_dft']['total_spectrum_normalized']
        amplitude_sino1 = metrics['interpolated_dft']['amplitude_sino1']
        total_spectrum = metrics['interpolated_dft']['total_spectrum']
        max_heta = metrics['interpolated_dft']['max_heta']

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

        if self.peaks_dir is not None:
            dump_numpy_variable(self.peaks_dir, 'found directions')
        else:
            print('No directions found !!!')

        print(f'estimations after direction refinement :')
        print(self.waves_fields_estimations)

        # Displays
        display_initial_data(self)
        display_radon_transforms(self)
        display_radon_transforms(self, refinement_phase=True)
        display_context(self)
