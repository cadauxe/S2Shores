# -*- coding: utf-8 -*-
"""
Class managing the computation of wave fields from two images taken at a small time interval.


:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 5 mars 2021

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
import os
from typing import TYPE_CHECKING # @NoMove

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np

from .sinogram_display import (build_sinogram_display, 
                               build_sinogram_difference_display,
                               build_sinogram_spectral_display,
                               build_correl_spectrum_matrix)
from .waves_image_display import (create_pseudorgb,
                                   build_display_waves_image,
                                   build_display_pseudorgb)
from .polar_display import build_polar_display
from .display_utils import get_display_title_with_kernel

if TYPE_CHECKING:
    from ..local_bathymetry.spatial_dft_bathy_estimator import (
        SpatialDFTBathyEstimator)  # @UnusedImport
	
def display_dft_sinograms(local_estimator: 'SpatialDFTBathyEstimator') -> None:
    # plt.close('all')
    nrows = 2
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
    fig.suptitle(get_display_title_with_kernel(local_estimator), fontsize=12)

    first_image = local_estimator.ortho_sequence[0]
    second_image = local_estimator.ortho_sequence[1]

    # First Plot line = Image1 Circle Filtered / pseudoRGB Circle Filtered/ Image2 Circle Filtered
    image1_circle_filtered = first_image.pixels * first_image.circle_image
    image2_circle_filtered = second_image.pixels * second_image.circle_image
    pseudo_rgb_circle_filtered = create_pseudorgb(image1_circle_filtered, image2_circle_filtered)

    build_display_waves_image(fig, axs[0, 0], 'Image1 Circle Filtered', image1_circle_filtered,
                              subplot_pos=[nrows, ncols, 1],
                              resolution=first_image.resolution, cmap='gray')
    build_display_pseudorgb(fig,
                            axs[0,
                                1],
                            'Pseudo RGB Circle Filtered',
                            pseudo_rgb_circle_filtered,
                            resolution=first_image.resolution,
                            subplot_pos=[nrows,
                                         ncols,
                                         2],
                            coordinates=False)
    build_display_waves_image(fig, axs[0, 2], 'Image2 Circle Filtered', image2_circle_filtered,
                              resolution=second_image.resolution,
                              subplot_pos=[nrows, ncols, 3], cmap='gray', coordinates=False)

    # Second Plot line = Sinogram1 / Sinogram2-Sinogram1 / Sinogram2
    first_radon_transform = local_estimator.radon_transforms[0]
    sinogram1, directions1 = first_radon_transform.get_as_arrays()
    second_radon_transform = local_estimator.radon_transforms[1]
    sinogram2, directions2 = second_radon_transform.get_as_arrays()
    radon_difference = (sinogram2 / np.max(np.abs(sinogram2))) - \
        (sinogram1 / np.max(np.abs(sinogram1)))

    # get main direction
    estimations = local_estimator.bathymetry_estimations
    sorted_estimations_args = estimations.argsort_on_attribute(
        local_estimator.final_estimations_sorting)
    main_direction = estimations.get_estimations_attribute('direction')[
        sorted_estimations_args[0]]

    plt_min = local_estimator.global_estimator.local_estimator_params['DEBUG']['PLOT_MIN']
    plt_max = local_estimator.global_estimator.local_estimator_params['DEBUG']['PLOT_MAX']

    build_sinogram_display(
        axs[1, 0], 'Sinogram1 [Radon Transform on Master Image]', sinogram1, directions1, sinogram2,
        main_direction, plt_min, plt_max)
    build_sinogram_difference_display(
        axs[1, 1], 'Sinogram2 - Sinogram1', radon_difference, directions2, plt_min, plt_max, cmap='bwr')
    build_sinogram_display(
        axs[1, 2], 'Sinogram2 [Radon Transform on Slave Image]', sinogram2, directions2, sinogram1,
        main_direction, plt_min, plt_max, ordonate=False)

    plt.tight_layout()
    point_id = f'{int(local_estimator.location.x)}_{int(local_estimator.location.y)}'

    plt.savefig(
        os.path.join(
            local_estimator.global_estimator.debug_path,
            'display_sinograms_debug_point_' +
            point_id +
            '_theta_' +
            f'{int(main_direction)}' +
            '.png'),
        dpi=300)
    dft_sino = plt.figure(2)
    return dft_sino

def display_dft_sinograms_spectral_analysis(
        local_estimator: 'SpatialDFTBathyEstimator') -> None:
    # plt.close('all')
    nrows = 3
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 15))
    fig.suptitle(get_display_title_with_kernel(local_estimator), fontsize=12)

    # First Plot line = Sinogram1 / Sinogram2-Sinogram1 / Sinogram2
    first_radon_transform = local_estimator.radon_transforms[0]
    sinogram1, directions1 = first_radon_transform.get_as_arrays()
    second_radon_transform = local_estimator.radon_transforms[1]
    sinogram2, directions2 = second_radon_transform.get_as_arrays()
    radon_difference = (sinogram2 / np.max(np.abs(sinogram2))) - \
        (sinogram1 / np.max(np.abs(sinogram1)))
    # get main direction
    estimations = local_estimator.bathymetry_estimations
    sorted_estimations_args = estimations.argsort_on_attribute(
        local_estimator.final_estimations_sorting)
    main_direction = estimations.get_estimations_attribute('direction')[
        sorted_estimations_args[0]]

    delta_time = estimations.get_estimations_attribute('delta_time')[
        sorted_estimations_args[0]]
    plt_min = local_estimator.global_estimator.local_estimator_params['DEBUG']['PLOT_MIN']
    plt_max = local_estimator.global_estimator.local_estimator_params['DEBUG']['PLOT_MAX']

    build_sinogram_display(
        axs[0, 0], 'Sinogram1 [Radon Transform on Master Image]',
        sinogram1, directions1, sinogram2, main_direction, plt_min, plt_max, abscissa=False)
    build_sinogram_difference_display(
        axs[0, 1], 'Sinogram2 - Sinogram1', radon_difference, directions2, plt_min, plt_max,
        abscissa=False, cmap='bwr')
    build_sinogram_display(
        axs[0, 2], 'Sinogram2 [Radon Transform on Slave Image]', sinogram2, directions2, sinogram1,
        main_direction, plt_min, plt_max, ordonate=False, abscissa=False)

    # Second Plot line = Spectral Amplitude of Sinogram1 [after DFT] / CSM Amplitude /
    # Spectral Amplitude of Sinogram2 [after DFT]

    sino1_fft = first_radon_transform.get_sinograms_standard_dfts()
    sino2_fft = second_radon_transform.get_sinograms_standard_dfts()
    kfft = local_estimator._metrics['kfft']

    build_sinogram_spectral_display(
        axs[1, 0], 'Spectral Amplitude Sinogram1 [DFT]',
        np.abs(sino1_fft), directions1, kfft, plt_min, plt_max, abscissa=False, cmap='cmc.oslo_r')
    build_correl_spectrum_matrix(
        axs[1, 1], local_estimator, sino1_fft, sino2_fft, kfft, plt_min, plt_max, 'amplitude',
        'Cross Spectral Matrix (Amplitude)')
    build_sinogram_spectral_display(axs[1, 2], 'Spectral Amplitude Sinogram2 [DFT]',
                                    np.abs(sino2_fft), directions2, kfft, plt_min, plt_max,
                                    ordonate=False, abscissa=False, cmap='cmc.oslo_r')

    csm_phase, spectrum_amplitude, sinograms_correlation_fft = \
        local_estimator._cross_correl_spectrum(sino1_fft, sino2_fft)

    # Third Plot line = Spectral Amplitude of Sinogram1 [after DFT] * CSM Phase /
    # CSM Amplitude * CSM Phase / Spectral Amplitude of Sinogram2 [after DFT] * CSM Phase

    build_sinogram_spectral_display(
        axs[2, 0], 'Spectral Amplitude Sinogram1 [DFT] * CSM_Phase',
        np.abs(sino1_fft) * csm_phase, directions1, kfft, plt_min, plt_max, abscissa=False, cmap='cmc.vik')
    build_correl_spectrum_matrix(
        axs[2, 1], local_estimator, sino1_fft, sino2_fft, kfft, plt_min, plt_max, 'phase',
        'Cross Spectral Matrix (Amplitude * Phase-shifts)')
    build_sinogram_spectral_display(
        axs[2, 2], 'Spectral Amplitude Sinogram2 [DFT] * CSM_Phase',
        np.abs(sino2_fft) * csm_phase, directions2, kfft, plt_min, plt_max,
        ordonate=False, abscissa=False, cmap='cmc.vik')
    plt.tight_layout()
    point_id = f'{int(local_estimator.location.x)}_{int(local_estimator.location.y)}'

    plt.savefig(
        os.path.join(
            local_estimator.global_estimator.debug_path,
            'display_sinograms_spectral_analysis_debug_point_' +
            point_id +
            '_theta_' +
            f'{int(main_direction)}' +
            '.png'),
        dpi=300)
    dft_sino_spectral = plt.figure(3)
    return dft_sino_spectral

def display_polar_images_dft(local_estimator: 'SpatialDFTBathyEstimator') -> None:
    # plt.close('all')
    nrows = 1
    ncols = 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6))
    fig.suptitle(get_display_title_with_kernel(local_estimator), fontsize=12)

    estimations = local_estimator.bathymetry_estimations
    sorted_estimations_args = estimations.argsort_on_attribute(
        local_estimator.final_estimations_sorting)
    main_direction = estimations.get_estimations_attribute('direction')[
        sorted_estimations_args[0]]
    ener_max = estimations.get_estimations_attribute('energy_ratio')[
        sorted_estimations_args[0]]
    main_wavelength = estimations.get_estimations_attribute('wavelength')[
        sorted_estimations_args[0]]
    delta_time = estimations.get_estimations_attribute('delta_time')[
        sorted_estimations_args[0]]
    dir_max_from_north = (270 - main_direction) % 360
    arrows = [(wfe.direction, wfe.energy_ratio) for wfe in estimations]

    print('ARROWS', arrows)
    first_image = local_estimator.ortho_sequence[0]

    # First Plot line = Image1 / pseudoRGB / Image2
    build_display_waves_image(
        fig,
        axs[0],
        'Image1 [Cartesian Projection]',
        first_image.original_pixels,
        resolution=first_image.resolution,
        subplot_pos=[
            nrows,
            ncols,
            1],
        directions=arrows,
        cmap='gray')

    first_radon_transform = local_estimator.radon_transforms[0]
    _, directions1 = first_radon_transform.get_as_arrays()
    second_radon_transform = local_estimator.radon_transforms[1]
    sino1_fft = first_radon_transform.get_sinograms_standard_dfts()
    sino2_fft = second_radon_transform.get_sinograms_standard_dfts()

    csm_phase, spectrum_amplitude, sinograms_correlation_fft = \
        local_estimator._cross_correl_spectrum(sino1_fft, sino2_fft)
    csm_amplitude = np.abs(sinograms_correlation_fft)

    # Retrieve arguments corresponding to the arrow with the maximum energy
    arrow_max = (dir_max_from_north, ener_max, main_wavelength)
    theta_id = f'{int(main_direction)}'

    print('-->ARROW SIGNING THE MAX ENERGY [DFN, ENERGY, WAVELENGTH]]=', arrow_max)
    polar = csm_amplitude * csm_phase

    # set negative values to 0 to avoid mirror display
    polar[polar < 0] = 0
    build_polar_display(
        fig,
        axs[1],
        'CSM Amplitude * CSM Phase-Shifts [Polar Projection]',
        local_estimator,
        polar,
        first_image.resolution,
        dir_max_from_north,
        main_wavelength,
        subplot_pos=[
            1,
            2,
            2])

    plt.tight_layout()
    point_id = f'{int(local_estimator.location.x)}_{int(local_estimator.location.y)}'

    plt.savefig(
        os.path.join(
            local_estimator.global_estimator.debug_path,
            'display_polar_images_debug_point_' + point_id + '_theta_' + theta_id + '.png'),
        dpi=300)
    polar_plot = plt.figure(4)
    return polar_plot

def display_waves_images_dft(local_estimator: 'SpatialDFTBathyEstimator') -> None:
    # plt.close('all')
    nrows = 3
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    fig.suptitle(get_display_title_with_kernel(local_estimator), fontsize=12)
    first_image = local_estimator.ortho_sequence[0]
    second_image = local_estimator.ortho_sequence[1]
    pseudo_rgb = create_pseudorgb(first_image.original_pixels, second_image.original_pixels)

    # First Plot line = Image1 / pseudoRGB / Image2
    build_display_waves_image(fig, axs[0, 0], 'Image1', first_image.original_pixels,
                              resolution=first_image.resolution,
                              subplot_pos=[nrows, ncols, 1], cmap='gray')
    build_display_pseudorgb(fig, axs[0, 1], 'Pseudo RGB', pseudo_rgb,
                            resolution=first_image.resolution,
                            subplot_pos=[nrows, ncols, 2], coordinates=False)
    build_display_waves_image(fig, axs[0, 2], 'Image2', second_image.original_pixels,
                              resolution=second_image.resolution,
                              subplot_pos=[nrows, ncols, 3], cmap='gray', coordinates=False)

    # Second Plot line = Image1 Filtered / pseudoRGB Filtered/ Image2 Filtered
    pseudo_rgb_filtered = create_pseudorgb(first_image.pixels, second_image.pixels)
    build_display_waves_image(fig, axs[1, 0], 'Image1 Filtered', first_image.pixels,
                              resolution=first_image.resolution,
                              subplot_pos=[nrows, ncols, 4], cmap='gray')
    build_display_pseudorgb(fig, axs[1, 1], 'Pseudo RGB Filtered', pseudo_rgb_filtered,
                            resolution=first_image.resolution,
                            subplot_pos=[nrows, ncols, 5], coordinates=False)
    build_display_waves_image(fig, axs[1, 2], 'Image2 Filtered', second_image.pixels,
                              resolution=second_image.resolution,
                              subplot_pos=[nrows, ncols, 6], cmap='gray', coordinates=False)

    # Third Plot line = Image1 Circle Filtered / pseudoRGB Circle Filtered/ Image2 Circle Filtered
    image1_circle_filtered = first_image.pixels * first_image.circle_image
    image2_circle_filtered = second_image.pixels * second_image.circle_image
    pseudo_rgb_circle_filtered = create_pseudorgb(image1_circle_filtered, image2_circle_filtered)
    build_display_waves_image(fig, axs[2, 0], 'Image1 Circle Filtered', image1_circle_filtered,
                              resolution=first_image.resolution,
                              subplot_pos=[nrows, ncols, 7], cmap='gray')
    build_display_pseudorgb(fig,
                            axs[2,
                                1],
                            'Pseudo RGB Circle Filtered',
                            pseudo_rgb_circle_filtered,
                            resolution=first_image.resolution,
                            subplot_pos=[nrows,
                                         ncols,
                                         8],
                            coordinates=False)
    build_display_waves_image(fig, axs[2, 2], 'Image2 Circle Filtered', image2_circle_filtered,
                              resolution=second_image.resolution,
                              subplot_pos=[nrows, ncols, 9], cmap='gray', coordinates=False)
    plt.tight_layout()
    point_id = f'{int(local_estimator.location.x)}_{int(local_estimator.location.y)}'

    estimations = local_estimator.bathymetry_estimations
    sorted_estimations_args = estimations.argsort_on_attribute(
        local_estimator.final_estimations_sorting)
    main_direction = estimations.get_estimations_attribute('direction')[
        sorted_estimations_args[0]]

    plt.savefig(
        os.path.join(
            local_estimator.global_estimator.debug_path,
            'display_waves_images_debug_point_' +
            point_id +
            '_theta_' +
            f'{int(main_direction)}' +
            '.png'),
        dpi=300)
    waves_image = plt.figure(1)
    return waves_image
