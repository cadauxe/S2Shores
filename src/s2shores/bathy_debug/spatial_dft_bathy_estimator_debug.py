# -*- coding: utf-8 -*-
""" Class for debugging the Spatial DFT estimator.

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 5 mars 2021
"""
import numpy as np
from matplotlib import pyplot as plt

from ..generic_utils.numpy_utils import dump_numpy_variable
from ..local_bathymetry.spatial_dft_bathy_estimator import \
    SpatialDFTBathyEstimator
from .local_bathy_estimator_debug import LocalBathyEstimatorDebug
from .wave_fields_display import (display_context, display_dft_sinograms,
                                  display_dft_sinograms_spectral_analysis,
                                  display_initial_data,
                                  display_polar_images_dft,
                                  display_radon_transforms,
                                  display_waves_images_dft)


class SpatialDFTBathyEstimatorDebug(LocalBathyEstimatorDebug, SpatialDFTBathyEstimator):
    """ Class allowing to debug the estimations made by a SpatialDFTBathyEstimator
    """

    def explore_results(self) -> None:

        self.print_variables()
        print('estimations after direction refinement, before physical constraint filtering and before sorting :')
        print(self.bathymetry_estimations)

        # Displays
        # display_initial_data(self)
        if len(self.bathymetry_estimations) >0:
            waves_image = display_waves_images_dft(self)
            dft_sinograms = display_dft_sinograms(self)
            dft_sino_spectral = display_dft_sinograms_spectral_analysis(self)
            polar_plot = display_polar_images_dft(self)
            plt.show()
        else:
            print('No estimation to display.')


    def print_variables(self) -> None:
        metrics = self.metrics

        initial_sino1_fft = self.radon_transforms[0].get_sinograms_standard_dfts()
        initial_total_spectrum_normalized = metrics['standard_dft']['total_spectrum_normalized']
        initial_phase_shift = np.angle(metrics['standard_dft']['sinograms_correlation_fft'])

        sino1_fft = self.radon_transforms[0].get_sinograms_interpolated_dfts()
        phase_shift = np.angle(metrics['interpolated_dft']['sinograms_correlation_fft'])
        total_spectrum_normalized = metrics['interpolated_dft']['total_spectrum_normalized']

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

        dump_numpy_variable(total_spectrum_normalized, 'refined total_spectrum_normalized')
