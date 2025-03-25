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
from typing import TYPE_CHECKING  # @NoMove

import matplotlib.pyplot as plt
import numpy as np

from ..image_processing.waves_radon import WavesRadon
from .sinogram_display import (build_sinogram_display, 
                               build_sinogram_difference_display,
                               build_sinogram_1D_display_master,
                               build_sinogram_1D_cross_correlation,
                               build_sinogram_1D_display_slave,
                               build_sinogram_2D_cross_correlation)
from .waves_image_display import (create_pseudorgb,
                                   build_display_waves_image,
                                   build_display_pseudorgb)
from .display_utils import get_display_title_with_kernel

if TYPE_CHECKING:
    from ..local_bathymetry.spatial_correlation_bathy_estimator import (
        SpatialCorrelationBathyEstimator)  # @UnusedImport
 
def display_sinograms_1D_analysis_spatial_correlation(
        local_estimator: 'SpatialCorrelationBathyEstimator') -> None:

    # plt.close('all')
    nrows = 3
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
    fig.suptitle(get_display_title_with_kernel(local_estimator), fontsize=12)

    first_image = local_estimator.ortho_sequence[0]
    second_image = local_estimator.ortho_sequence[1]

    # First Plot line = Sinogram1 / Sinogram2-Sinogram1 / Sinogram2
    first_radon_transform = WavesRadon(first_image)
    sinogram1, directions1 = first_radon_transform.get_as_arrays()
    second_radon_transform = WavesRadon(second_image)
    sinogram2, directions2 = second_radon_transform.get_as_arrays()
    radon_difference = (sinogram2 / np.max(np.abs(sinogram2))) - \
        (sinogram1 / np.max(np.abs(sinogram1)))
    # get main direction
    main_direction = local_estimator.bathymetry_estimations.get_estimations_attribute('direction')[
        0]

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

    # Second Plot line = SINO_1 [1D along estimated direction] / Cross-Correlation Signal /
    # SINO_2 [1D along estimated direction resulting from Image1]
    # Check if the main direction belongs to the plotting interval [plt_min:plt_max]
    if main_direction < plt_min or main_direction > plt_max:
        theta_label = main_direction % (-np.sign(main_direction) * 180.0)
    else:
        theta_label = main_direction
    title_sino1 = '[Master Image] Sinogram 1D along $\Theta$={:.1f}° '.format(theta_label)
    title_sino2 = '[Slave Image] Sinogram 1D'.format(theta_label)
    correl_mode = local_estimator.global_estimator.local_estimator_params['CORRELATION_MODE']

    build_sinogram_1D_display_master(
        axs[1, 0], title_sino1, sinogram1, directions1, main_direction, plt_min, plt_max)
    build_sinogram_1D_cross_correlation(
        axs[1, 1], 'Normalized Cross-Correlation Signal', sinogram1, directions1, main_direction,
        sinogram2, directions2, plt_min, plt_max, correl_mode, ordonate=False)
    build_sinogram_1D_display_slave(
        axs[1, 2], title_sino2,
        sinogram2, directions2, main_direction, plt_min, plt_max, ordonate=False)

    # Third Plot line = Image [2D] Cross correl Sino1[main dir] with Sino2 all directions /
    # Image [2D] of Cross correlation 1D between SINO1 & SINO 2 for each direction /
    # Image [2D] Cross correl Sino2[main dir] with Sino1 all directions
    # Check if the main direction belongs to the plotting interval [plt_min:plt_ramax]

    title_cross_correl1 = 'Normalized Cross-Correlation Signal between \n Sino1[$\Theta$={:.1f}°] and Sino2[All Directions]'.format(
        theta_label)
    title_cross_correl2 = 'Normalized Cross-Correlation Signal between \n Sino2[$\Theta$={:.1f}°] and Sino1[All Directions]'.format(
        0)
    title_cross_correl_2D = '2D-Normalized Cross-Correlation Signal between \n Sino1 and Sino2 for Each Direction'

    build_sinogram_2D_cross_correlation(
        axs[2, 0], title_cross_correl1, sinogram1, directions1, main_direction,
        sinogram2, plt_min, plt_max, correl_mode, choice='one_dir', imgtype='master')
    build_sinogram_2D_cross_correlation(
        axs[2, 1], title_cross_correl_2D, sinogram1, directions1, main_direction,
        sinogram2, plt_min, plt_max, correl_mode, choice='all_dir', imgtype='master', ordonate=False)
    build_sinogram_2D_cross_correlation(
        axs[2, 2], title_cross_correl2, sinogram2, directions2, main_direction,
        sinogram1, plt_min, plt_max, correl_mode, choice='one_dir', imgtype='slave', ordonate=False)

    plt.tight_layout()
    point_id = f'{int(local_estimator.location.x)}_{int(local_estimator.location.y)}'
    theta_id = f'{int(main_direction)}'

    plt.savefig(
        os.path.join(
            local_estimator.global_estimator.debug_path,
            'display_sinograms_1D_analysis_debug_point_' +
            point_id +
            '_theta_' +
            theta_id +
            '.png'),
        dpi=300)
    # plt.show()
    dft_sino_spectral = plt.figure(3)
    return dft_sino_spectral

def display_sinograms_spatial_correlation(
        local_estimator: 'SpatialCorrelationBathyEstimator') -> None:
    # plt.close('all')
    nrows = 2
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
    fig.suptitle(get_display_title_with_kernel(local_estimator), fontsize=12)
    first_image = local_estimator.ortho_sequence[0]
    second_image = local_estimator.ortho_sequence[1]

    # Since wfe.energy_ratio not available for SpatialCorrelation:
    default_arrow_length = np.shape(first_image.original_pixels)[0]
    arrows = [(wfe.direction, default_arrow_length)
              for wfe in local_estimator.bathymetry_estimations]

    # First Plot line = Image1 Circle Filtered / pseudoRGB Circle Filtered/ Image2 Circle Filtered
    image1_circle_filtered = first_image.pixels * first_image.circle_image
    image2_circle_filtered = second_image.pixels * second_image.circle_image
    pseudo_rgb_circle_filtered = create_pseudorgb(image1_circle_filtered, image2_circle_filtered)
    build_display_waves_image(fig, axs[0, 0], 'Master Image Circle Filtered',
                              image1_circle_filtered, subplot_pos=[nrows, ncols, 1],
                              resolution=first_image.resolution, directions=arrows, cmap='gray')
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
    build_display_waves_image(fig,
                              axs[0,
                                  2],
                              'Slave Image Circle Filtered',
                              image2_circle_filtered,
                              resolution=second_image.resolution,
                              subplot_pos=[nrows,
                                           ncols,
                                           3],
                              directions=arrows,
                              cmap='gray',
                              coordinates=False)

    # Second Plot line = Sinogram1 / Sinogram2-Sinogram1 / Sinogram2
    first_radon_transform = WavesRadon(first_image)
    sinogram1, directions1 = first_radon_transform.get_as_arrays()
    second_radon_transform = WavesRadon(second_image)
    sinogram2, directions2 = second_radon_transform.get_as_arrays()
    radon_difference = (sinogram2 / np.max(np.abs(sinogram2))) - \
        (sinogram1 / np.max(np.abs(sinogram1)))
    # get main direction
    main_direction = local_estimator.bathymetry_estimations.get_estimations_attribute('direction')[
        0]

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

    theta_id = f'{int(main_direction)}'

    plt.savefig(
        os.path.join(
            local_estimator.global_estimator.debug_path,
            'display_sinograms_debug_point_' + point_id + '_theta_' + theta_id + '.png'),
        dpi=300)
    # plt.show()
    dft_sino = plt.figure(2)
    return dft_sino

def display_waves_images_spatial_correl(
        local_estimator: 'SpatialCorrelationBathyEstimation') -> None:
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

    main_dir = local_estimator.bathymetry_estimations.get_estimations_attribute('direction')[0]

    theta_id = f'{int(main_dir)}'
    plt.savefig(
        os.path.join(
            local_estimator.global_estimator.debug_path,
            'display_waves_images_debug_point_' + point_id + '_theta_' + theta_id + '.png'),
        dpi=300)
    # plt.show()
    waves_image = plt.figure(1)
    return waves_image
