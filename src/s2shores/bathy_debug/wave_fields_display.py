# -*- coding: utf-8 -*-
"""
Class managing the computation of wave fields from two images taken at a small time interval.


:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 5 mars 2021
"""
import math
import os
from typing import TYPE_CHECKING, List, Optional, Tuple  # @NoMove

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy as scp
import scipy.ndimage.filters as filters
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure

from s2shores.data_model.wave_field_sample_geometry import \
    WaveFieldSampleGeometry
from s2shores.generic_utils.image_utils import (cross_correlation,
                                                normalized_cross_correlation)
from s2shores.image_processing.waves_radon import WavesRadon

from ..bathy_physics import wavenumber_offshore

if TYPE_CHECKING:
    from ..local_bathymetry.spatial_dft_bathy_estimator \
        import SpatialDFTBathyEstimator  # @UnusedImport
    from ..local_bathymetry.spatial_correlation_bathy_estimator \
        import SpatialCorrelationBathyEstimator  # @UnusedImport


def display_curve(data: np.ndarray, legend: str) -> None:
    _, ax = plt.subplots()
    ax.plot(data)
    ax.set_title(legend)
    plt.show()


def display_3curves(data1: np.ndarray, data2: np.ndarray, data3: np.ndarray) -> None:
    _, ax = plt.subplots(3)
    ax[0].plot(data1)
    ax[1].plot(data2)
    ax[2].plot(data3)
    plt.show()


def display_4curves(data1: np.ndarray, data2: np.ndarray, data3: np.ndarray,
                    data4: np.ndarray) -> None:
    _, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].plot(data1)
    ax[1, 0].plot(data2)
    ax[0, 1].plot(data3)
    ax[1, 1].plot(data4)
    plt.show()


def display_image(data: np.ndarray, legend: str) -> None:
    _, ax = plt.subplots()
    ax.imshow(data, aspect='auto', cmap='gray')
    ax.set_title(legend)
    plt.show()


def get_display_title(local_estimator: 'SpatialDFTBathyEstimator') -> str:
    title = f'{local_estimator.global_estimator._ortho_stack.short_name} {local_estimator.location}'
    return title


def build_image_display(axes: Axes, title: str, image: np.ndarray,
                        directions: Optional[List[Tuple[float, float]]] = None,
                        cmap: Optional[str] = None) -> None:
    imin = np.min(image)
    imax = np.max(image)
    axes.imshow(image, norm=Normalize(vmin=imin, vmax=imax), cmap=cmap)
    (l1, l2) = np.shape(image)
    coeff_length_max = np.max((list(zip(*directions))[1])) + 1
    radius = np.floor(min(l1, l2) / 2) - 1
    if directions is not None:
        for direction, coeff_length in directions:
            arrow_length = radius * coeff_length / coeff_length_max
            dir_rad = np.deg2rad(direction)
            axes.arrow(l1 // 2, l2 // 2,
                       np.cos(dir_rad) * arrow_length, -np.sin(dir_rad) * arrow_length,
                       head_width=2, head_length=3, color='r')
    axes.set_title(title)


def build_directional_2d_display(axes: Axes, title: str, values: np.ndarray,
                                 directions: np.ndarray, **kwargs: dict) -> None:
    extent = [np.min(directions), np.max(directions), 0, values.shape[0]]
    imin = np.min(values)
    imax = np.max(values)
    axes.imshow(values, norm=Normalize(vmin=imin, vmax=imax), extent=extent, **kwargs)
    axes.set_xticks(directions[::40])
    plt.setp(axes.get_xticklabels(), fontsize=8)
    axes.set_title(title, fontsize=10)


def build_directional_curve_display(axes: Axes, title: str,
                                    values: np.ndarray, directions: np.ndarray) -> None:
    axes.plot(directions, values)
    axes.set_xticks(directions[::20])
    plt.setp(axes.get_xticklabels(), fontsize=8)
    axes.set_title(title)


def display_initial_data(local_estimator: 'SpatialDFTBathyEstimator') -> None:
    plt.close('all')
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    fig.suptitle(get_display_title(local_estimator), fontsize=12)
    arrows = [(wfe.direction, wfe.energy_ratio) for wfe in local_estimator.bathymetry_estimations]
    first_image = local_estimator.ortho_sequence[0]
    second_image = local_estimator.ortho_sequence[1]
    build_image_display(axs[0, 0], 'first image original', first_image.original_pixels,
                        directions=arrows, cmap='gray')
    build_image_display(axs[1, 0], 'second image original', second_image.original_pixels,
                        directions=arrows, cmap='gray')
    build_image_display(axs[0, 1], 'first image filtered', first_image.pixels,
                        directions=arrows, cmap='gray')
    build_image_display(axs[1, 1], 'second image filtered', second_image.pixels,
                        directions=arrows, cmap='gray')
    first_radon_transform = local_estimator.radon_transforms[0]
    second_radon_transform = local_estimator.radon_transforms[1]

    values, directions = first_radon_transform.get_as_arrays()
    build_directional_2d_display(axs[0, 2], 'first radon transform', values, directions)
    values, directions = second_radon_transform.get_as_arrays()
    build_directional_2d_display(axs[1, 2], 'second radon transform', values, directions)
    plt.show()


def get_display_title_with_kernel(local_estimator: 'SpatialDFTBathyEstimator') -> str:
    title = f'{local_estimator.global_estimator._ortho_stack.short_name} {local_estimator.location}'
    smooth_kernel_xsize = local_estimator.global_estimator.smoothing_lines_size
    smooth_kernel_ysize = local_estimator.global_estimator.smoothing_columns_size
    filter_info = ''
    if smooth_kernel_xsize == 0 and smooth_kernel_ysize == 0:
        filter_info = f' (i.e. Smoothing Filter DEACTIVATED!)'

    return title + \
        f'\n Smoothing Kernel Size = [{2 * smooth_kernel_xsize + 1}px*{2 * smooth_kernel_ysize + 1}px]' + filter_info


def create_pseudorgb(image1: np.ndarray, image2: np.ndarray,) -> np.ndarray:
    normalized_im2 = (image2 - image2.min()) / (image2.max() - image2.min())
    normalized_im1 = (image1 - image1.min()) / (image1.max() - image1.min())

    ps_rgb = np.dstack((normalized_im2, normalized_im1, normalized_im2))
    ps_rgb = ps_rgb - ps_rgb.min()
    return ps_rgb / (ps_rgb.max() - ps_rgb.min())


def build_display_pseudorgb(fig: Figure, axes: Axes, title: str, image: np.ndarray,
                            resolution: float,
                            subplot_pos: [float, float, float],
                            directions: Optional[List[Tuple[float, float]]] = None,
                            cmap: Optional[str] = None, coordinates: bool=True) -> None:

    (l1, l2, l3) = np.shape(image)
    imin = np.min(image)
    imax = np.max(image)
    #imsh = axes.imshow(image, norm=Normalize(vmin=imin, vmax=imax))
    #axes.imshow(image, norm=Normalize(vmin=imin, vmax=imax))
    axes.imshow(image, norm=Normalize(vmin=imin, vmax=imax), cmap=cmap)
    # create polar axes in the foreground and remove its background to see through
    subplot_locator = int(f'{subplot_pos[0]}{subplot_pos[1]}{subplot_pos[2]}')
    ax_polar = fig.add_subplot(subplot_locator, polar=True)
    ax_polar.set_yticklabels([])
    polar_ticks = np.arange(4) * np.pi / 2.
    polar_labels = ['0°', '90°', '180°', '-90°']

    plt.xticks(polar_ticks, polar_labels, size=9, color='blue')
    for i, label in enumerate(ax_polar.get_xticklabels()):
        label.set_rotation(i * 45)
    ax_polar.set_facecolor("None")

    xmax = f'{l1}px \n {np.round((l1-1)*resolution)}m'
    axes.set_xticks([0, l1 - 1], ['0', xmax], fontsize=8)
    ymax = f'{l2}px \n {np.round((l2-1)*resolution)}m'

    if coordinates:
        axes.set_yticks([0, l2 - 1], [ymax, '0'], fontsize=8)
    else:
        axes.set_yticks([0, l2 - 1], ['', ''], fontsize=8)
        axes.set_xticks([0, l1 - 1], ['\n', ' \n'], fontsize=8)

    if directions is not None:
        # Normalization of arrows length
        coeff_length_max = np.max((list(zip(*directions))[1]))
        radius = np.floor(min(l1, l2) / 2) - 5
        for direction, coeff_length in directions:
            arrow_length = radius * coeff_length / coeff_length_max
            dir_rad = np.deg2rad(direction) + np.pi
            axes.arrow(l1 // 2, l2 // 2,
                       np.cos(dir_rad) * arrow_length, -np.sin(dir_rad) * arrow_length,
                       head_width=2, head_length=3, color='r')

    axes.xaxis.tick_top()
    axes.set_title(title, fontsize=9, loc='center')
    #fig.colorbar(imsh, ax=axes, location='right', shrink=1.0)
    # Manage blank spaces
    # plt.tight_layout()


def build_display_waves_image(fig: Figure, axes: Axes, title: str, image: np.ndarray,
                              resolution: float,
                              subplot_pos: [float, float, float],
                              directions: Optional[List[Tuple[float, float]]] = None,
                              cmap: Optional[str] = None, coordinates: bool=True) -> None:

    (l1, l2) = np.shape(image)
    imin = np.min(image)
    imax = np.max(image)
    #imsh = axes.imshow(image, norm=Normalize(vmin=imin, vmax=imax), cmap=cmap)
    axes.imshow(image, norm=Normalize(vmin=imin, vmax=imax), cmap=cmap)
    # create polar axes in the foreground and remove its background to see through
    subplot_locator = int(f'{subplot_pos[0]}{subplot_pos[1]}{subplot_pos[2]}')
    ax_polar = fig.add_subplot(subplot_locator, polar=True)
    ax_polar.set_yticklabels([])
    polar_ticks = np.arange(4) * np.pi / 2.
    polar_labels = ['0°', '90°', '180°', '-90°']

    plt.xticks(polar_ticks, polar_labels, size=9, color='blue')
    for i, label in enumerate(ax_polar.get_xticklabels()):
        label.set_rotation(i * 45)
    ax_polar.set_facecolor("None")

    xmax = f'{l1}px \n {np.round((l1-1)*resolution)}m'
    axes.set_xticks([0, l1 - 1], ['0', xmax], fontsize=8)
    ymax = f'{l2}px \n {np.round((l2-1)*resolution)}m'
    #axes.set_yticks([0, l2 - 1], [ymax, '0'], fontsize=8)
    if coordinates:
        axes.set_yticks([0, l2 - 1], [ymax, '0'], fontsize=8)
    else:
        axes.set_yticks([0, l2 - 1], ['', ''], fontsize=8)
        axes.set_xticks([0, l1 - 1], ['\n', ' \n'], fontsize=8)

    if directions is not None:
        # Normalization of arrows length
        coeff_length_max = np.max((list(zip(*directions))[1]))
        radius = np.floor(min(l1, l2) / 2) - 5
        for direction, coeff_length in directions:
            arrow_length = radius * coeff_length / coeff_length_max
            dir_rad = np.deg2rad(direction)  # + np.pi
            axes.arrow(l1 // 2, l2 // 2,
                       -np.sin(dir_rad) * arrow_length, np.cos(dir_rad) * arrow_length,
                       head_width=2, head_length=3, color='r')

    axes.xaxis.tick_top()
    axes.set_title(title, fontsize=9, loc='center')
    # Manage blank spaces
    # plt.tight_layout()


def display_waves_images_dft(local_estimator: 'SpatialDFTBathyEstimator') -> None:
    # plt.close('all')
    nrows = 3
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    fig.suptitle(get_display_title_with_kernel(local_estimator), fontsize=12)
    #arrows = [(wfe.direction, wfe.energy_ratio) for wfe in local_estimator.bathymetry_estimations]
    first_image = local_estimator.ortho_sequence[0]
    #spatial_ref = local_estimator.global_estimator._ortho_stack.build_spatial_ref()
    #epsg_code = local_estimator.global_estimator._ortho_stack.epsg_code
    #up_left = local_estimator.global_estimator._ortho_stack._upper_left_corner
    #low_right = local_estimator.global_estimator._ortho_stack._lower_right_corner
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
    build_display_pseudorgb(fig, axs[2, 1], 'Pseudo RGB Circle Filtered', pseudo_rgb_circle_filtered,
                            resolution=first_image.resolution,
                            subplot_pos=[nrows, ncols, 8], coordinates=False)
    build_display_waves_image(fig, axs[2, 2], 'Image2 Circle Filtered', image2_circle_filtered,
                              resolution=second_image.resolution,
                              subplot_pos=[nrows, ncols, 9], cmap='gray', coordinates=False)
    plt.tight_layout()
    point_id = f'{np.int(local_estimator.location.x)}_{np.int(local_estimator.location.y)}'

    main_dir = local_estimator._bathymetry_estimations.get_estimations_attribute('direction')[
        0]
    theta_id = f'{np.int(main_dir)}'

    plt.savefig(
        os.path.join(
            local_estimator.global_estimator._debug_path,
            "display_waves_images_debug_point_" + point_id + "_theta_" + theta_id + ".png"),
        dpi=300)
    waves_image = plt.figure(1)
    return waves_image


def display_waves_images_spatial_correl(
        local_estimator: 'SpatialCorrelationBathyEstimation') -> None:
    # plt.close('all')
    nrows = 3
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    fig.suptitle(get_display_title_with_kernel(local_estimator), fontsize=12)

    #arrows = [(wfe.direction, wfe.energy_ratio) for wfe in local_estimator.bathymetry_estimations]
    first_image = local_estimator.ortho_sequence[0]
    second_image = local_estimator.ortho_sequence[1]
    pseudo_rgb = create_pseudorgb(first_image.original_pixels, second_image.original_pixels)

    # Since wfe.eneergy_ratio not available for SpatialCorrelation:
    #default_arrow_length = np.shape(first_image.original_pixels)[0]
    # arrows = [(wfe.direction, default_arrow_length)
    #          for wfe in local_estimator.bathymetry_estimations]

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
    build_display_pseudorgb(fig, axs[2, 1], 'Pseudo RGB Circle Filtered', pseudo_rgb_circle_filtered,
                            resolution=first_image.resolution,
                            subplot_pos=[nrows, ncols, 8], coordinates=False)
    build_display_waves_image(fig, axs[2, 2], 'Image2 Circle Filtered', image2_circle_filtered,
                              resolution=second_image.resolution,
                              subplot_pos=[nrows, ncols, 9], cmap='gray', coordinates=False)
    plt.tight_layout()
    point_id = f'{np.int(local_estimator.location.x)}_{np.int(local_estimator.location.y)}'
    main_dir = local_estimator._bathymetry_estimations.get_estimations_attribute('direction')[
        0]
    theta_id = f'{np.int(main_dir)}'
    plt.savefig(
        os.path.join(
            local_estimator.global_estimator._debug_path,
            "display_waves_images_debug_point_" + point_id + "_theta_" + theta_id + ".png"),
        dpi=300)
    # plt.show()
    waves_image = plt.figure(1)
    return waves_image


def build_sinogram_display(axes: Axes, title: str, values1: np.ndarray, directions: np.ndarray,
                           values2: np.ndarray, main_theta: float, plt_rng: float,
                           ordonate: bool=True, abscissa: bool=True, master: bool=True,
                           **kwargs: dict) -> None:
    extent = [np.min(directions), np.max(directions),
              np.floor(-values1.shape[0] / 2),
              np.ceil(values1.shape[0] / 2)]
    axes.imshow(values1, aspect='auto', extent=extent, **kwargs)
    orig_main_theta = main_theta
    normalized_var1 = (np.var(values1, axis=0) /
                       np.max(np.var(values1, axis=0)) - 0.5) * values1.shape[0]
    normalized_var2 = (np.var(values2, axis=0) /
                       np.max(np.var(values2, axis=0)) - 0.5) * values2.shape[0]
    axes.plot(directions, normalized_var2,
              color="red", lw=1, ls='--', label='Normalized Variance \n Comparative Sinogram')
    axes.plot(directions, normalized_var1,
              color="white", lw=0.8, label='Normalized Variance \n Reference Sinogram')

    pos1 = np.where(normalized_var1 == np.max(normalized_var1))

    # Check coherence of main direction between Master / Slave
    if directions[pos1][0] * main_theta < 0:
        main_theta = directions[pos1][0] % (np.sign(main_theta) * 180.0)
    # Check if the main direction belongs to the plotting interval [-plt_range:plt_range]
    if main_theta < -plt_rng or main_theta > plt_rng:
        main_theta %= -np.sign(main_theta) * 180.0
    theta_label = '$\Theta${:.1f}° [Variance Max]'.format(main_theta)
    theta_label_orig = '$\Theta${:.1f}° [Main Direction]'.format(orig_main_theta)

    axes.axvline(main_theta, np.floor(-values1.shape[0] / 2), np.ceil(values1.shape[0] / 2),
                 color='orange', ls='--', lw=1, label=theta_label)

    axes.axvline(orig_main_theta, np.floor(-values1.shape[0] / 2), np.ceil(values1.shape[0] / 2),
                 color='blue', ls='--', lw=1, label=theta_label_orig)

    legend = axes.legend(loc='upper right', shadow=True, fontsize=6)
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('C0')
    axes.grid(lw=0.5, color='white', alpha=0.7, linestyle='-')
    axes.set_xlim(-plt_rng, plt_rng)
    axes.set_xticks(np.arange(-plt_rng, plt_rng + 1, 45))
    plt.setp(axes.get_xticklabels(), fontsize=8)

    if ordonate:
        axes.set_ylabel(r'$\rho$ [pixels]', fontsize=8)
    else:
        axes.yaxis.set_ticklabels([])
    if abscissa:
        axes.set_xlabel(r'Direction Angle $\theta$ [degrees]', fontsize=8)

    axes.set_title(title, fontsize=10)
    axes.tick_params(axis='both', which='major', labelsize=8)


def build_sinogram_difference_display(axes: Axes, title: str, values: np.ndarray,
                                      directions: np.ndarray, plt_rng: float,
                                      abscissa: bool=True, cmap: Optional[str] = None,
                                      **kwargs: dict) -> None:

    extent = [np.min(directions), np.max(directions),
              np.floor(-values.shape[0] / 2),
              np.ceil(values.shape[0] / 2)]

    axes.imshow(values, cmap=cmap, aspect='auto', extent=extent, **kwargs)

    axes.grid(lw=0.5, color='black', alpha=0.7, linestyle='-')
    axes.yaxis.set_ticklabels([])
    axes.set_xlim(-plt_rng, plt_rng)
    axes.set_xticks(np.arange(-plt_rng, plt_rng + 1, 45))
    plt.setp(axes.get_xticklabels(), fontsize=8)

    if abscissa:
        axes.set_xlabel(r'Direction Angle $\theta$ [degrees]', fontsize=8)

    axes.set_title(title, fontsize=10)
    axes.tick_params(axis='both', which='major', labelsize=8)


def display_dft_sinograms(local_estimator: 'SpatialDFTBathyEstimator') -> None:
    # plt.close('all')
    nrows = 2
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
    fig.suptitle(get_display_title_with_kernel(local_estimator), fontsize=12)

    #arrows = [(wfe.direction, wfe.energy_ratio) for wfe in local_estimator.bathymetry_estimations]
    first_image = local_estimator.ortho_sequence[0]
    second_image = local_estimator.ortho_sequence[1]

    # First Plot line = Image1 Circle Filtered / pseudoRGB Circle Filtered/ Image2 Circle Filtered
    image1_circle_filtered = first_image.pixels * first_image.circle_image
    image2_circle_filtered = second_image.pixels * second_image.circle_image
    pseudo_rgb_circle_filtered = create_pseudorgb(image1_circle_filtered, image2_circle_filtered)

    # According to Delta_Time sign, proceed with arrow's direction inversion
    delta_time = local_estimator._bathymetry_estimations.get_estimations_attribute('delta_time')[0]
    delta_phase = local_estimator._bathymetry_estimations.get_estimations_attribute(
        'delta_phase')[0]
    arrows = [(wfe.direction, wfe.energy_ratio) for wfe in local_estimator.bathymetry_estimations]

    corrected_arrows = []
    arrows_from_north = []
    for arrow_dir, arrow_ener in arrows:
        arrow_dir_from_north = (270 - arrow_dir)
        arrows_from_north.append((arrow_dir_from_north, arrow_ener))
        arrows = arrows_from_north
    print(' ARROW DIRECTIONS FROM NORTH =', arrows)

    if np.sign(delta_time * delta_phase) < 0:
        print('Display_polar_images_dft: inversion of arrows direction!!!!!!')
        for arrow_dir, arrow_ener in arrows:
            arrow_dir %= 180
            corrected_arrows.append((arrow_dir_from_north, arrow_ener))
            arrows = corrected_arrows

    build_display_waves_image(fig, axs[0, 0], 'Image1 Circle Filtered', image1_circle_filtered,
                              subplot_pos=[nrows, ncols, 1],
                              resolution=first_image.resolution, cmap='gray')
    build_display_pseudorgb(fig, axs[0, 1], 'Pseudo RGB Circle Filtered', pseudo_rgb_circle_filtered,
                            resolution=first_image.resolution,
                            subplot_pos=[nrows, ncols, 2], coordinates=False)
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
    main_direction = local_estimator._bathymetry_estimations.get_estimations_attribute('direction')[
        0]
    plt_range = local_estimator.global_estimator.local_estimator_params['TUNING']['PLOT_RANGE']

    build_sinogram_display(
        axs[1, 0], 'Sinogram1 [Radon Transform on Master Image]', sinogram1, directions1, sinogram2,
        main_direction, plt_range)
    build_sinogram_difference_display(
        axs[1, 1], 'Sinogram2 - Sinogram1', radon_difference, directions2, plt_range, cmap='bwr')
    build_sinogram_display(
        axs[1, 2], 'Sinogram2 [Radon Transform on Slave Image]', sinogram2, directions2, sinogram1,
        main_direction, plt_range, ordonate=False)

    plt.tight_layout()
    point_id = f'{np.int(local_estimator.location.x)}_{np.int(local_estimator.location.y)}'
    main_dir = local_estimator._bathymetry_estimations.get_estimations_attribute('direction')[
        0]
    theta_id = f'{np.int(main_dir)}'

    plt.savefig(
        os.path.join(
            local_estimator.global_estimator._debug_path,
            "display_sinograms_debug_point_" + point_id + "_theta_" + theta_id + ".png"),
        dpi=300)
    dft_sino = plt.figure(2)
    return dft_sino


def display_sinograms_spatial_correlation(
        local_estimator: 'SpatialCorrelationBathyEstimator') -> None:
    # plt.close('all')
    nrows = 2
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
    fig.suptitle(get_display_title_with_kernel(local_estimator), fontsize=12)
    first_image = local_estimator.ortho_sequence[0]
    second_image = local_estimator.ortho_sequence[1]

    # Since wfe.eneergy_ratio not available for SpatialCorrelation:
    default_arrow_length = np.shape(first_image.original_pixels)[0]
    arrows = [(wfe.direction, default_arrow_length)
              for wfe in local_estimator.bathymetry_estimations]

    # According to Delta_Time sign, proceed with arrow's direction inversion
    delta_time = local_estimator._bathymetry_estimations.get_estimations_attribute('delta_time')[0]
    #print('DELTA TIME', delta_time)
    corrected_arrows = []
    if np.sign(delta_time) < 0:
        print('Display_polar_images_dft: inversion of arrows direction!')
        for arrow_dir, arrow_ener in arrows:
            arrow_dir %= 180
            corrected_arrows.append((arrow_dir, arrow_ener))
            arrows = corrected_arrows

    # First Plot line = Image1 Circle Filtered / pseudoRGB Circle Filtered/ Image2 Circle Filtered
    image1_circle_filtered = first_image.pixels * first_image.circle_image
    image2_circle_filtered = second_image.pixels * second_image.circle_image
    pseudo_rgb_circle_filtered = create_pseudorgb(image1_circle_filtered, image2_circle_filtered)
    build_display_waves_image(fig, axs[0, 0], 'Master Image Circle Filtered', image1_circle_filtered,
                              subplot_pos=[nrows, ncols, 1],
                              resolution=first_image.resolution, cmap='gray')
    build_display_pseudorgb(fig, axs[0, 1], 'Pseudo RGB Circle Filtered', pseudo_rgb_circle_filtered,
                            resolution=first_image.resolution,
                            subplot_pos=[nrows, ncols, 2], coordinates=False)
    build_display_waves_image(fig, axs[0, 2], 'Slave Image Circle Filtered', image2_circle_filtered,
                              resolution=second_image.resolution,
                              subplot_pos=[nrows, ncols, 3], cmap='gray', coordinates=False)

    # Second Plot line = Sinogram1 / Sinogram2-Sinogram1 / Sinogram2
    first_radon_transform = WavesRadon(first_image)
    sinogram1, directions1 = first_radon_transform.get_as_arrays()
    second_radon_transform = WavesRadon(second_image)
    sinogram2, directions2 = second_radon_transform.get_as_arrays()
    radon_difference = (sinogram2 / np.max(np.abs(sinogram2))) - \
        (sinogram1 / np.max(np.abs(sinogram1)))
    # get main direction
    main_direction = local_estimator._bathymetry_estimations.get_estimations_attribute('direction')[
        0]
    plt_range = local_estimator.global_estimator.local_estimator_params['TUNING']['PLOT_RANGE']

    build_sinogram_display(
        axs[1, 0], 'Sinogram1 [Radon Transform on Master Image]', sinogram1, directions1, sinogram2,
        main_direction, plt_range)
    build_sinogram_difference_display(
        axs[1, 1], 'Sinogram2 - Sinogram1', radon_difference, directions2, plt_range, cmap='bwr')
    build_sinogram_display(
        axs[1, 2], 'Sinogram2 [Radon Transform on Slave Image]', sinogram2, directions2, sinogram1,
        main_direction, plt_range, ordonate=False)

    plt.tight_layout()
    point_id = f'{np.int(local_estimator.location.x)}_{np.int(local_estimator.location.y)}'
    main_dir = local_estimator._bathymetry_estimations.get_estimations_attribute('direction')[
        0]
    theta_id = f'{np.int(main_dir)}'

    plt.savefig(
        os.path.join(
            local_estimator.global_estimator._debug_path,
            "display_sinograms_debug_point_" + point_id + "_theta_" + theta_id + ".png"),
        dpi=300)
    # plt.show()
    dft_sino = plt.figure(2)
    return dft_sino


def build_sinogram_spectral_display(axes: Axes, title: str, values: np.ndarray,
                                    directions: np.ndarray, kfft: np.ndarray, plt_rng: float,
                                    ordonate: bool=True, abscissa: bool=True, **kwargs: dict) -> None:
    extent = [np.min(directions), np.max(directions), 0.0, kfft.max()]
    axes.imshow(values, aspect='auto', origin="lower", extent=extent, **kwargs)
    # axes.plot(directions, ((np.var(values, axis=0) / np.max(np.var(values, axis=0))) * kfft.max()),
    #          color="white", lw=0.7)
    axes.plot(directions, ((np.max(values, axis=0) / np.max(np.max(values, axis=0))) * kfft.max()),
              color="white", lw=0.7, label='Normalized Maximum')
    legend = axes.legend(loc='upper right', shadow=True, fontsize=6)
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('C0')

    axes.grid(lw=0.5, color='white', alpha=0.7, linestyle='-')
    axes.set_xlim(-plt_rng, plt_rng)
    axes.set_xticks(np.arange(-plt_rng, plt_rng + 1, 45))
    plt.setp(axes.get_xticklabels(), fontsize=8)

    if ordonate:
        axes.set_ylabel(r'Wavenumber $\nu$ [m$^{-1}$]', fontsize=8)
    else:
        axes.yaxis.set_ticklabels([])
    if abscissa:
        axes.set_xlabel(r'Direction Angle $\theta$ [degrees]', fontsize=8)

    axes.set_title(title, fontsize=10)
    axes.tick_params(axis='both', which='major', labelsize=8)


def build_sinogram_fft_display(axes: Axes, title: str, values: np.ndarray, directions: np.ndarray,
                               kfft: np.ndarray, plt_rng: float, type: str,
                               ordonate: bool=True, abscissa: bool=True, **kwargs: dict) -> None:

    extent = [np.min(directions), np.max(directions), 0.0, kfft.max()]
    axes.imshow(values, aspect='auto', origin="lower", extent=extent, **kwargs)
    #loc_transect = np.where(directions == -29.0)
    #transect = values[:, loc_transect[0]]
    #print('TRANSECT = ', transect)
    #neighborhood_size = 10
    #val_max = filters.maximum_filter(values, neighborhood_size)
    #val_min = filters.minimum_filter(values, neighborhood_size)
    #axes.imshow(val_min, aspect='auto', origin="lower", extent=extent, **kwargs)

    if type == 'amplitude':
        axes.plot(directions, ((np.var(values, axis=0) / np.max(np.var(values, axis=0))) * kfft.max()),
                  color="white", lw=0.7, label='Normalized Variance')
        axes.plot(directions, ((np.max(values, axis=0) / np.max(np.max(values, axis=0))) * kfft.max()),
                  color="orange", lw=0.7, label='Normalized Maximum')
        legend = axes.legend(loc='upper right', shadow=True, fontsize=6)
        # Put a nicer background color on the legend.
        legend.get_frame().set_facecolor('C0')
    axes.grid(lw=0.5, color='white', alpha=0.7, linestyle='-')
    axes.set_xlim(-plt_rng, plt_rng)
    axes.set_xticks(np.arange(-plt_rng, plt_rng + 1, 45))
    plt.setp(axes.get_xticklabels(), fontsize=8)

    if ordonate:
        axes.set_ylabel(r'Wavenumber $\nu$ [m$^{-1}$]', fontsize=8)
    else:
        axes.yaxis.set_ticklabels([])
    if abscissa:
        axes.set_xlabel(r'Direction Angle $\theta$ [degrees]', fontsize=8)

    axes.set_title(title, fontsize=10)
    axes.tick_params(axis='both', which='major', labelsize=8)


def build_correl_spectrum_matrix(axes: Axes, local_estimator: 'SpatialDFTBathyEstimator',
                                 sino1_fft: np.ndarray, sino2_fft: np.ndarray, kfft: np.ndarray,
                                 plt_rng: float, type: str, title: str,
                                 refinement_phase: bool=False) -> None:
    radon_transform = local_estimator.radon_transforms[0]
    if not refinement_phase:
        _, directions = radon_transform.get_as_arrays()
    else:
        directions = radon_transform.directions_interpolated_dft
    #sinograms_correlation_fft = metrics[key]['sinograms_correlation_fft']
    # equals sinograms_correlation_fft from
    # local_estimator._cross_correl_spectrum(sino1_fft, sino2_fft)
    #total_spectrum = metrics[key]['total_spectrum']
    #total_spectrum_normalized = metrics[key]['total_spectrum_normalized']
    #max_heta = metrics[key]['max_heta']
    csm_phase, spectrum_amplitude, sinograms_correlation_fft = \
        local_estimator._cross_correl_spectrum(sino1_fft, sino2_fft)

    # EB method == sinograms_correlation_fft from
    # local_estimator._cross_correl_spectrum(sino1_fft, sino2_fft)
    #cross_spectral_matrix = sino2_fft * np.conj(sino1_fft)
    #csm_amplitude = np.abs(cross_spectral_matrix)
    csm_amplitude = np.abs(sinograms_correlation_fft)

    if type == 'amplitude':
        build_sinogram_fft_display(axes, title, csm_amplitude, directions, kfft, plt_rng,
                                   type, ordonate=False, abscissa=False)
    if type == 'phase':
        build_sinogram_fft_display(axes, title, csm_amplitude * csm_phase, directions, kfft, plt_rng,
                                   type, ordonate=False, abscissa=False)
    delta_time = local_estimator._bathymetry_estimations.get_estimations_attribute('delta_time')[0]
    if type == 'phase_corrected':
        build_sinogram_fft_display(axes, title, csm_amplitude * csm_phase * np.sign(delta_time),
                                   directions, kfft, plt_rng, type, ordonate=False)


def display_dft_sinograms_spectral_analysis(
        local_estimator: 'SpatialDFTBathyEstimator') -> None:
    # plt.close('all')
    nrows = 4
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
    main_direction = local_estimator._bathymetry_estimations.get_estimations_attribute('direction')[
        0]
    delta_time = local_estimator._bathymetry_estimations.get_estimations_attribute('delta_time')[0]
    plt_range = local_estimator.global_estimator.local_estimator_params['TUNING']['PLOT_RANGE']

    build_sinogram_display(
        axs[0, 0], 'Sinogram1 [Radon Transform on Master Image]',
        sinogram1, directions1, sinogram2, main_direction, plt_range, abscissa=False)
    build_sinogram_difference_display(
        axs[0, 1], 'Sinogram2 - Sinogram1', radon_difference, directions2, plt_range,
        abscissa=False, cmap='bwr')
    build_sinogram_display(
        axs[0, 2], 'Sinogram2 [Radon Transform on Slave Image]', sinogram2, directions2, sinogram1,
        main_direction, plt_range, ordonate=False, abscissa=False)

    # Second Plot line = Spectral Amplitude of Sinogram1 [after DFT] / CSM Amplitude /
    # Spectral Amplitude of Sinogram2 [after DFT]

    sino1_fft = first_radon_transform.get_sinograms_standard_dfts()
    sino2_fft = second_radon_transform.get_sinograms_standard_dfts()
    kfft = local_estimator._metrics['kfft']

    plt_range = local_estimator.global_estimator.local_estimator_params['TUNING']['PLOT_RANGE']

    build_sinogram_spectral_display(
        axs[1, 0], 'Spectral Amplitude Sinogram1 [DFT]',
        np.abs(sino1_fft), directions1, kfft, plt_range, abscissa=False)
    build_correl_spectrum_matrix(
        axs[1, 1], local_estimator, sino1_fft, sino2_fft, kfft, plt_range, 'amplitude',
        'Cross Spectral Matrix (Amplitude)')
    build_sinogram_spectral_display(
        axs[1, 2], 'Spectral Amplitude Sinogram2 [DFT]',
        np.abs(sino2_fft), directions2, kfft, plt_range, ordonate=False, abscissa=False)

    # Third Plot line = Spectral Amplitude of Sinogram1 [after DFT] * CSM Phase /
    # CSM Amplitude * CSM Phase / Spectral Amplitude of Sinogram2 [after DFT] * CSM Phase
    # Manage blank spaces"
    csm_phase, spectrum_amplitude, sinograms_correlation_fft = \
        local_estimator._cross_correl_spectrum(sino1_fft, sino2_fft)
    build_sinogram_spectral_display(
        axs[2, 0], 'Spectral Amplitude Sinogram1 [DFT] * CSM_Phase',
        np.abs(sino1_fft) * csm_phase, directions1, kfft, plt_range, abscissa=False)
    build_correl_spectrum_matrix(
        axs[2, 1], local_estimator, sino1_fft, sino2_fft, kfft, plt_range, 'phase',
        'Cross Spectral Matrix (Amplitude * Phase-shifts)')
    build_sinogram_spectral_display(
        axs[2, 2], 'Spectral Amplitude Sinogram2 [DFT] * CSM_Phase',
        np.abs(sino2_fft) * csm_phase, directions2, kfft, plt_range, ordonate=False, abscissa=False)

    # Add Cross Spectral Matrix display according to the Delta_Time sign
    build_sinogram_spectral_display(
        axs[3, 0], 'Same Graph as above with $\Delta$t sign correction',
        np.abs(sino1_fft) * csm_phase * np.sign(delta_time), directions1, kfft, plt_range)
    build_correl_spectrum_matrix(
        axs[3, 1], local_estimator, sino1_fft, sino2_fft, kfft, plt_range, 'phase_corrected',
        'Same Graph as above with $\Delta$t sign correction')
    build_sinogram_spectral_display(
        axs[3, 2], 'Same Graph as above with $\Delta$t sign correction',
        np.abs(sino2_fft) * csm_phase * np.sign(delta_time), directions2, kfft, plt_range, ordonate=False)
    plt.tight_layout()
    point_id = f'{np.int(local_estimator.location.x)}_{np.int(local_estimator.location.y)}'
    main_dir = local_estimator._bathymetry_estimations.get_estimations_attribute('direction')[
        0]
    theta_id = f'{np.int(main_dir)}'

    plt.savefig(
        os.path.join(
            local_estimator.global_estimator._debug_path,
            "display_sinograms_spectral_analysis_debug_point_" + point_id + "_theta_" + theta_id + ".png"),
        dpi=300)
    dft_sino_spectral = plt.figure(3)
    return dft_sino_spectral


def build_correl_spectrum_matrix_spatial_correlation(axes: Axes, local_estimator: 'SpatialCorrelationBathyEstimator',
                                                     sino1_fft: np.ndarray, sino2_fft: np.ndarray, kfft: np.ndarray,
                                                     type: str, title: str, refinement_phase: bool=False) -> None:
    """ Computes the cross correlation spectrum of the radon transforms of the images, possibly
        restricted to a limited set of directions.

        :param ilocal_estimator
        :param sino1_fft: the DFT of the first sinogram, either standard or interpolated
        :param sino2_fft: the DFT of the second sinogram, either standard or interpolated
        :returns: A tuple of 2 numpy arrays and a dictionary with:
                  - the phase shifts
                  - the spectrum amplitude
                  - a dictionary containing intermediate results for debugging purposes
    """
    radon_transform = WavesRadon(local_estimator.ortho_sequence[0])
    if not refinement_phase:
        _, directions = radon_transform.get_as_arrays()
    else:
        directions = radon_transform.directions_interpolated_dft

    sinograms_correlation_fft = sino2_fft * np.conj(sino1_fft)
    csm_phase = np.angle(sinograms_correlation_fft)
    csm_amplitude = np.abs(sinograms_correlation_fft)

    if type == 'amplitude':
        build_sinogram_fft_display(axes, title, csm_amplitude, directions, kfft,
                                   type, ordonate=False, abscissa=False)
    if type == 'phase':
        build_sinogram_fft_display(axes, title, csm_amplitude * csm_phase, directions, kfft,
                                   type, ordonate=False)


def build_sinogram_1D_display_master(axes: Axes, title: str, values1: np.ndarray, directions: np.ndarray,
                                     main_theta: float, plt_rng: float,
                                     ordonate: bool=True, abscissa: bool=True, **kwargs: dict) -> None:
    index_theta = np.int(main_theta - np.min(directions))
    # Check if the main direction belongs to the plotting interval [-plt_range:plt_range]
    if main_theta < -plt_rng or main_theta > plt_rng:
        main_theta %= -np.sign(main_theta) * 180.0
    theta_label = 'Sinogram 1D along \n$\Theta$={:.1f}°'.format(main_theta)
    nb_pixels = np.shape(values1[:, index_theta])[0]
    absc = np.arange(-nb_pixels / 2, nb_pixels / 2)
    axes.plot(absc, np.flip((values1[:, index_theta] / np.max(np.abs(values1[:, index_theta])))),
              color="orange", lw=0.8, label=theta_label)

    legend = axes.legend(loc='upper right', shadow=True, fontsize=6)
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('C0')
    axes.grid(lw=0.5, color='gray', alpha=0.7, linestyle='-')
    axes.set_xticks(np.arange(ceil_to_nearest_10(-nb_pixels / 2),
                              floor_to_nearest_10(nb_pixels / 2 + 10), 25))
    axes.set_yticks(np.arange(-1, 1.2, 0.25))
    plt.setp(axes.get_xticklabels(), fontsize=8)

    if ordonate:
        axes.set_ylabel(r'Normalized Sinogram Amplitude', fontsize=8)
    else:
        axes.yaxis.set_ticklabels([])
    if abscissa:
        axes.set_xlabel(r'$\rho$ [pixels]', fontsize=8)

    axes.set_title(title, fontsize=10)
    axes.tick_params(axis='both', which='major', labelsize=8)


def build_sinogram_1D_display_slave(axes: Axes, title: str, values: np.ndarray, directions: np.ndarray,
                                    main_theta: float, plt_rng: float,
                                    ordonate: bool=True, abscissa: bool=True, **kwargs: dict) -> None:

    normalized_var = (np.var(values, axis=0) /
                      np.max(np.var(values, axis=0)) - 0.5) * values.shape[0]
    pos = np.where(normalized_var == np.max(normalized_var))
    main_theta_slave = directions[pos][0]

    # Check coherence of main direction between Master / Slave
    if directions[pos][0] * main_theta < 0:
        main_theta_slave = directions[pos][0] % (np.sign(main_theta) * 180.0)

    index_theta_master = np.int(main_theta - np.min(directions))
    index_theta_slave = np.int(main_theta_slave - np.min(directions))

    # Check if the main direction belongs to the plotting interval [-plt_range:plt_range]
    if main_theta < -plt_rng or main_theta > plt_rng:
        main_theta %= -np.sign(main_theta) * 180.0
    if main_theta_slave < -plt_rng or main_theta_slave > plt_rng:
        main_theta_slave %= -np.sign(main_theta_slave) * 180.0
    theta_label_master = 'along Master Main Direction\n$\Theta$={:.1f}°'.format(main_theta)
    theta_label_slave = 'along Slave Main Direction\n$\Theta$={:.1f}°'.format(main_theta_slave)
    nb_pixels = np.shape(values[:, index_theta_master])[0]
    absc = np.arange(-nb_pixels / 2, nb_pixels / 2)
    axes.plot(absc, np.flip((values[:, index_theta_master] / np.max(np.abs(values[:, index_theta_master])))),
              color="orange", lw=0.8, label=theta_label_master)
    axes.plot(absc, np.flip((values[:, index_theta_slave] / np.max(np.abs(values[:, index_theta_slave])))),
              color="blue", lw=0.8, ls='--', label=theta_label_slave)

    legend = axes.legend(loc='upper right', shadow=True, fontsize=6)
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('C0')
    axes.grid(lw=0.5, color='gray', alpha=0.7, linestyle='-')
    axes.set_xticks(np.arange(ceil_to_nearest_10(-nb_pixels / 2),
                              floor_to_nearest_10(nb_pixels / 2 + 10), 25))
    axes.set_yticks(np.arange(-1, 1.2, 0.25))
    plt.setp(axes.get_xticklabels(), fontsize=8)

    if ordonate:
        axes.set_ylabel(r'Normalized Sinogram Amplitude', fontsize=8)
    else:
        axes.yaxis.set_ticklabels([])
    if abscissa:
        axes.set_xlabel(r'$\rho$ [pixels]', fontsize=8)

    axes.set_title(title, fontsize=10)
    axes.tick_params(axis='both', which='major', labelsize=8)


def build_sinogram_1D_cross_correlation(axes: Axes, title: str, values1: np.ndarray,
                                        directions1: np.ndarray, main_theta: float,
                                        values2: np.ndarray, directions2: np.ndarray,
                                        plt_rng: float, correl_mode: str, ordonate: bool=True,
                                        abscissa: bool=True, **kwargs: dict) -> None:

    normalized_var = (np.var(values2, axis=0) /
                      np.max(np.var(values2, axis=0)) - 0.5) * values2.shape[0]
    pos2 = np.where(normalized_var == np.max(normalized_var))
    main_theta_slave = directions2[pos2][0]
    # Check coherence of main direction between Master / Slave
    if directions2[pos2][0] * main_theta < 0:
        main_theta_slave = directions2[pos2][0] % (np.sign(main_theta) * 180.0)

    index_theta1 = np.int(main_theta - np.min(directions1))
    # get 1D-sinogram1 along relevant direction
    sino1_1D = values1[:, index_theta1]
    # theta_label1 = 'Sinogram1 1D'  # along \n$\Theta$={:.1f}°'.format(main_theta)
    nb_pixels1 = np.shape(values1[:, index_theta1])[0]
    absc = np.arange(-nb_pixels1 / 2, nb_pixels1 / 2)
    # axes.plot(absc, np.flip((values1[:, index_theta1] / np.max(np.abs(values1[:, index_theta1])))),
    #          color="orange", lw=0.8, label=theta_label1)

    index_theta2_master = np.int(main_theta - np.min(directions2))
    index_theta2_slave = np.int(main_theta_slave - np.min(directions2))

    # get 1D-sinogram2 along relevant direction
    sino2_1D_master = values2[:, index_theta2_master]
    # theta_label2_master = 'Sinogram2 1D MASTER'  # along \n$\Theta$={:.1f}°'.format(main_theta)
    #nb_pixels2 = np.shape(values2[:, index_theta2_master])[0]
    #absc2 = np.arange(-nb_pixels2 / 2, nb_pixels2 / 2)
    # axes.plot(absc2, np.flip((values2[:, index_theta2_master] / np.max(np.abs(values2[:, index_theta2_master])))),
    #          color="black", lw=0.8, ls='--', label=theta_label2_master)

    # Check if the main direction belongs to the plotting interval [-plt_range:plt_range]
    if main_theta < -plt_rng or main_theta > plt_rng:
        main_theta_label = main_theta % (-np.sign(main_theta) * 180.0)
    else:
        main_theta_label = main_theta
    if main_theta_slave < -plt_rng or main_theta_slave > plt_rng:
        main_theta_slave_label = main_theta_slave % (-np.sign(main_theta_slave) * 180.0)
    else:
        main_theta_slave_label = main_theta_slave

    # Compute Cross-Correlation between Sino1 [Master Man Direction] & Sino2 [Master Main Direction]
    sino_cross_corr_norm_master = normalized_cross_correlation(
        np.flip(sino1_1D), np.flip(sino2_1D_master), correl_mode)
    label_correl_master = 'Sino1_1D[$\Theta$={:.1f}°] vs Sino2_1D[$\Theta$={:.1f}°]'.format(
        main_theta_label, main_theta_label)
    axes.plot(absc, sino_cross_corr_norm_master, color="red", lw=0.8, label=label_correl_master)

    sino2_1D_slave = values2[:, index_theta2_slave]
    # theta_label2_slave = 'Sinogram2 1D SLAVE'  # along \n$\Theta$={:.1f}°'.format(main_theta)
    # axes.plot(absc2, np.flip((values2[:, index_theta2_slave] / np.max(np.abs(values2[:, index_theta2_slave])))),
    #          color="green", lw=0.8, ls='--', label=theta_label2_slave)
    # Compute Cross-Correlation between Sino1 [Master Main Direction& Sino2 [Slave Main Direction]
    sino_cross_corr_norm_slave = normalized_cross_correlation(
        np.flip(sino1_1D), np.flip(sino2_1D_slave), correl_mode)

    label_correl_slave = 'Sino1_1D[$\Theta$={:.1f}°] vs Sino2_1D[$\Theta$={:.1f}°]'.format(
        main_theta_label, main_theta_slave_label)
    axes.plot(absc, sino_cross_corr_norm_slave, color="black", ls='--', lw=0.8,
              label=label_correl_slave)

    legend = axes.legend(loc='lower left', shadow=True, fontsize=6)
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('C0')
    axes.grid(lw=0.5, color='gray', alpha=0.7, linestyle='-')
    axes.set_xticks(np.arange(ceil_to_nearest_10(-nb_pixels1 / 2),
                              floor_to_nearest_10(nb_pixels1 / 2 + 10), 25))
    axes.set_yticks(np.arange(-1, 1.2, 0.25))
    plt.setp(axes.get_xticklabels(), fontsize=8)

    if ordonate:
        axes.set_ylabel(r'Normalized Sinogram Amplitude', fontsize=8)
    else:
        axes.yaxis.set_ticklabels([])
    if abscissa:
        axes.set_xlabel(r'$\rho$ [pixels]', fontsize=8)

    axes.set_title(title, fontsize=10)
    axes.tick_params(axis='both', which='major', labelsize=8)


def build_sinogram_2D_cross_correlation(axes: Axes, title: str, values1: np.ndarray,
                                        directions1: np.ndarray, main_theta: float,
                                        values2: np.ndarray, plt_rng: float, correl_mode: str,
                                        choice: str, imgtype: str, ordonate: bool=True,
                                        abscissa: bool=True, cmap: Optional[str] = None,
                                        **kwargs: dict) -> None:

    extent = [np.min(directions1), np.max(directions1),
              np.floor(-values1.shape[0] / 2),
              np.ceil(values1.shape[0] / 2)]

    if imgtype == 'slave':
        normalized_var = (np.var(values1, axis=0) /
                          np.max(np.var(values1, axis=0)) - 0.5) * values1.shape[0]
        pos = np.where(normalized_var == np.max(normalized_var))

        # Check coherence of main direction between Master / Slave
        if directions1[pos][0] * main_theta < 0:
            main_theta = directions1[pos][0] % (np.sign(main_theta) * 180.0)
        # Check if the main direction belongs to the plotting interval [-plt_range:plt_range]
        if main_theta < -plt_rng or main_theta > plt_rng:
            main_theta_label = main_theta % (-np.sign(main_theta) * 180.0)
        else:
            main_theta_label = main_theta

        title = 'Normalized Cross-Correlation Signal between \n Sino2[$\Theta$={:.1f}°] and Sino1[All Directions]'.format(
            main_theta_label)

    if choice == 'one_dir':
        index_theta1 = np.int(main_theta - np.min(directions1))
        # get 1D-sinogram1 along relevant direction
        sino1_1D = values1[:, index_theta1]
        # Proceed with 1D-Correlation between Sino1(main_dir) and Sino2(all_dir)
        values3 = np.transpose(values2).copy()
        index = 0

        for sino2_1D in zip(*values2):
            norm_cross_correl = normalized_cross_correlation(sino1_1D, sino2_1D, correl_mode)
            values3[index] = norm_cross_correl
            index += 1

        pos_max = np.where(np.transpose(values3) == np.max(np.transpose(values3)))

    if choice == 'all_dir':
        # get 1D-sinogram1 along relevant direction
        sino1_2D = np.transpose(values1)
        # Proceed with 1D-Correlation between Sino1(main_dir) and Sino2(all_dir)
        values3 = np.transpose(values2).copy()
        index = 0

        for sino2_1D in zip(*values2):
            norm_cross_correl = normalized_cross_correlation(sino1_2D[index], sino2_1D, correl_mode)
            values3[index] = norm_cross_correl
            index += 1

        # Compute variance associated to np.transpose(values3)
        normalized_var_val3 = (np.var(np.transpose(values3), axis=0) /
                               np.max(np.var(np.transpose(values3), axis=0)) - 0.5) * np.transpose(values3).shape[0]

        axes.plot(directions1, normalized_var_val3,
                  color="white", lw=1, ls='--', label='Normalized Variance', zorder=5)

        # Find position of the local maximum of the normalized variance of values3
        pos_val3 = np.where(normalized_var_val3 == np.max(normalized_var_val3))
        max_var_pos = directions1[pos_val3][0]

        # Check coherence of main direction between Master / Slave
        if directions1[pos_val3][0] * main_theta < 0:
            max_var_pos = directions1[pos_val3][0] % (np.sign(main_theta) * 180.0)
        # Check if the main direction belongs to the plotting interval [-plt_range:plt_range]
        if max_var_pos < -plt_rng or max_var_pos > plt_rng:
            max_var_pos %= -np.sign(max_var_pos) * 180.0

        max_var_label = '$\Theta$={:.1f}° [Variance Max]'.format(max_var_pos)
        axes.axvline(max_var_pos, np.floor(-values1.shape[0] / 2), np.ceil(values1.shape[0] / 2),
                     color='red', ls='--', lw=1, label=max_var_label, zorder=10)

    # Main 2D-plot
    axes.imshow(np.transpose(values3), cmap=cmap, aspect='auto', extent=extent, **kwargs)

    # Check if the main direction belongs to the plotting interval [-plt_range:plt_range]
    if main_theta < -plt_rng or main_theta > plt_rng:
        main_theta %= -np.sign(main_theta) * 180.0
    theta_label = '$\Theta$={:.1f}°'.format(main_theta)
    axes.axvline(main_theta, np.floor(-values1.shape[0] / 2), np.ceil(values1.shape[0] / 2),
                 color='orange', ls='--', lw=1, label=theta_label)

    legend = axes.legend(loc='upper right', shadow=True, fontsize=6)
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('C0')

    axes.grid(lw=0.5, color='black', alpha=0.7, linestyle='-')
    axes.set_xlim(-plt_rng, plt_rng)
    axes.set_xticks(np.arange(-plt_rng, plt_rng + 1, 45))
    plt.setp(axes.get_xticklabels(), fontsize=8)

    if choice == 'one_dir':
        xmax = directions1[pos_max[1]]
        ymax = np.floor(values1.shape[0] / 2) - pos_max[0]
        axes.scatter(xmax, ymax, c='r', s=20)
        notation = 'Local Maximum \n [$\Theta$={:.1f}°]'.format(xmax[0])
        axes.annotate(notation, xy=(xmax, ymax), xytext=(xmax + 10, ymax + 10), color='red')

    if ordonate:
        axes.set_ylabel(r'$\rho$ [pixels]', fontsize=8)
    else:
        axes.yaxis.set_ticklabels([])
    if abscissa:
        axes.set_xlabel(r'Direction Angle $\theta$ [degrees]', fontsize=8)

    axes.set_title(title, fontsize=10)
    axes.tick_params(axis='both', which='major', labelsize=8)


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
    main_direction = local_estimator._bathymetry_estimations.get_estimations_attribute('direction')[
        0]
    plt_range = local_estimator.global_estimator.local_estimator_params['TUNING']['PLOT_RANGE']

    build_sinogram_display(
        axs[0, 0], 'Sinogram1 [Radon Transform on Master Image]',
        sinogram1, directions1, sinogram2, main_direction, plt_range, abscissa=False)
    build_sinogram_difference_display(
        axs[0, 1], 'Sinogram2 - Sinogram1', radon_difference, directions2, plt_range,
        abscissa=False, cmap='bwr')
    build_sinogram_display(
        axs[0, 2], 'Sinogram2 [Radon Transform on Slave Image]', sinogram2, directions2, sinogram1,
        main_direction, plt_range, ordonate=False, abscissa=False)

    # Second Plot line = SINO_1 [1D along estimated direction] / Cross-Correlation Signal /
    # SINO_2 [1D along estimated direction resulting from Image1]
    # Check if the main direction belongs to the plotting interval [-plt_range:plt_range]
    if main_direction < -plt_range or main_direction > plt_range:
        theta_label = main_direction % (-np.sign(main_direction) * 180.0)
    else:
        theta_label = main_direction
    title_sino1 = '[Master Image] Sinogram 1D along $\Theta$={:.1f}° '.format(theta_label)
    title_sino2 = '[Slave Image] Sinogram 1D'.format(theta_label)
    correl_mode = local_estimator.global_estimator.local_estimator_params['CORRELATION_MODE']

    plt_range = local_estimator.global_estimator.local_estimator_params['TUNING']['PLOT_RANGE']

    build_sinogram_1D_display_master(
        axs[1, 0], title_sino1, sinogram1, directions1, main_direction, plt_range)
    build_sinogram_1D_cross_correlation(
        axs[1, 1], 'Normalized Cross-Correlation Signal', sinogram1, directions1, main_direction,
        sinogram2, directions2, plt_range, correl_mode, ordonate=False)
    build_sinogram_1D_display_slave(
        axs[1, 2], title_sino2,
        sinogram2, directions2, main_direction, plt_range, ordonate=False)

    # Third Plot line = Image [2D] Cross correl Sino1[main dir] with Sino2 all directions /
    # Image [2D] of Cross correlation 1D between SINO1 & SINO 2 for each direction /
    # Image [2D] Cross correl Sino2[main dir] with Sino1 all directions
    # Check if the main direction belongs to the plotting interval [-plt_range:plt_range]
    if main_direction < -plt_range or main_direction > plt_range:
        main_theta_label = main_direction % (-np.sign(main_direction) * 180.0)
    else:
        main_theta_label = main_direction
    title_cross_correl1 = 'Normalized Cross-Correlation Signal between \n Sino1[$\Theta$={:.1f}°] and Sino2[All Directions]'.format(
        main_theta_label)
    title_cross_correl2 = 'Normalized Cross-Correlation Signal between \n Sino2[$\Theta$={:.1f}°] and Sino1[All Directions]'.format(
        main_theta_label)
    title_cross_correl_2D = '2D-Normalized Cross-Correlation Signal between \n Sino1 and Sino2 for Each Direction'

    plt_range = local_estimator.global_estimator.local_estimator_params['TUNING']['PLOT_RANGE']

    build_sinogram_2D_cross_correlation(
        axs[2, 0], title_cross_correl1, sinogram1, directions1, main_direction,
        sinogram2, plt_range, correl_mode, choice='one_dir', imgtype='master')
    build_sinogram_2D_cross_correlation(
        axs[2, 1], title_cross_correl_2D, sinogram1, directions1, main_direction,
        sinogram2, plt_range, correl_mode, choice='all_dir', imgtype='master', ordonate=False)
    build_sinogram_2D_cross_correlation(
        axs[2, 2], title_cross_correl2, sinogram2, directions2, main_direction,
        sinogram1, plt_range, correl_mode, choice='one_dir', imgtype='slave', ordonate=False)

    plt.tight_layout()
    point_id = f'{np.int(local_estimator.location.x)}_{np.int(local_estimator.location.y)}'
    main_dir = local_estimator._bathymetry_estimations.get_estimations_attribute('direction')[
        0]
    theta_id = f'{np.int(main_dir)}'

    plt.savefig(
        os.path.join(
            local_estimator.global_estimator._debug_path,
            "display_sinograms_1D_analysis_debug_point_" + point_id + "_theta_" + theta_id + ".png"),
        dpi=300)
    # plt.show()
    dft_sino_spectral = plt.figure(3)
    return dft_sino_spectral


def build_polar_display(fig: Figure, axes: Axes, title: str,
                        local_estimator: 'SpatialDFTBathyEstimator',
                        values: np.ndarray, resolution: float,
                        subplot_pos: [float, float, float],
                        refinement_phase: bool=False, **kwargs: dict) -> None:

    radon_transform = local_estimator.radon_transforms[0]
    if not refinement_phase:
        _, directions = radon_transform.get_as_arrays()
    else:
        directions = radon_transform.directions_interpolated_dft

    # define wavenumbers according to image resolution
    Fs = 1 / resolution
    nb_wavenumbers = radon_transform.get_as_arrays()[0].shape[0]
    wavenumbers = np.arange(0, Fs / 2, Fs / nb_wavenumbers)

    # create polar axes in the foreground and remove its background to see through
    subplot_locator = int(f'{subplot_pos[0]}{subplot_pos[1]}{subplot_pos[2]}')
    ax_polar = plt.subplot(subplot_locator, polar=True)
    polar_ticks = np.arange(8) * np.pi / 4.
    # Xticks labels definition with 0° positioning to North with clockwise rotation
    polar_labels = ['90°', '45°', '0°', '315°', '270°', '225°', '180°', '135°']

    plt.xticks(polar_ticks, polar_labels, size=9, color='black')
    for i, label in enumerate(ax_polar.get_xticklabels()):
        label.set_rotation(i * 45)

    # get relevant attributes
    main_direction = local_estimator._bathymetry_estimations.get_estimations_attribute('direction')[
        0]
    main_wavelength = local_estimator._bathymetry_estimations.get_estimations_attribute(
        'wavelength')[0]
    direc_from_north = local_estimator._bathymetry_estimations.get_estimations_attribute(
        'direction_from_north')[0]
    delta_time = local_estimator._bathymetry_estimations.get_estimations_attribute(
        'delta_time')[0]
    delta_phase = local_estimator._bathymetry_estimations.get_estimations_attribute(
        'delta_phase')[0]

    # Constrains the Wavenumber plotting interval according to wavelength limitation set to 50m
    ax_polar.set_ylim(0, 0.02)
    requested_labels = np.array([500, 200, 100, 50, 25, main_wavelength]).round(2)
    requested_labels = np.flip(np.sort(requested_labels))
    rticks = 1 / requested_labels

    # Main information display
    print('MAIN DIRECTION', main_direction)
    print('DIRECTION FROM NORTH', direc_from_north)
    print('DELTA TIME', delta_time)
    print('DELTA PHASE', delta_phase)

    ax_polar.plot(np.radians((main_direction % 180)), 1 / main_wavelength, '*', color='black')

    ax_polar.annotate('Peak at \n[$\Theta$={:.1f}°, $\lambda$={:.2f}m]'.format((direc_from_north), main_wavelength),
                      xy=[np.radians(main_direction % 180), (1 / main_wavelength)],  # theta, radius
                      xytext=(0.5, 0.65),    # fraction, fraction
                      textcoords='figure fraction',
                      horizontalalignment='left',
                      verticalalignment='bottom',
                      fontsize=10, color='blue')

    # ax_polar.text(np.radians(main_direction), (1 / main_wavelength) * 1.25, r'Peak Wavelength $\lambda$ = {main_wavelength} [m]',
    #              rotation=0, ha='center', va='center', color='green')
    #rticks = np.arange(0.0, 0.11, 0.01)[1:]
    # Convert Wavenumber ticks into Wavelength ones
    #ax_polar.set_rgrids(rticks, labels=(1.0 / rticks).round(2), fontsize=12, angle=180, color='red')
    ax_polar.set_rgrids(rticks, labels=requested_labels, fontsize=12, angle=180, color='red')
    ax_polar.text(np.radians(50), ax_polar.get_rmax() * 1.25, r'Wavelength $\lambda$ [m]',
                  rotation=0, ha='center', va='center', color='red')
    ax_polar.set_rlabel_position(70)            # Moves the tick-labels
    ax_polar.set_rorigin(0)
    ax_polar.tick_params(axis='both', which='major', labelrotation=0, labelsize=8)
    ax_polar.grid(linewidth=0.5)

    # Define background color
    norm = TwoSlopeNorm(vcenter=1, vmin=0, vmax=3)
    ax_polar.set_facecolor(plt.cm.bwr_r(norm(3.0)))

    # Values to be plotted
    plotval = np.abs(values) / np.max(np.abs(values))

    # Add the last element of the list to the list.
    # This is necessary or the line from 330 deg to 0 degree does not join up on the plot.
    directions = np.append(directions, directions[0])
    plotval = np.concatenate((plotval, plotval[:, 0].reshape(plotval.shape[0], 1)), axis=1)

    ax_polar.contourf(np.deg2rad(directions), wavenumbers, plotval, cmap="gist_ncar")
    ax_polar.set_title(title, fontsize=9, loc='center')

    axes.xaxis.tick_top()
    axes.set_aspect('equal')
    # Manage blank spaces
    # plt.tight_layout()


def display_polar_images_dft(local_estimator: 'SpatialDFTBathyEstimator') -> None:
    # plt.close('all')
    nrows = 1
    ncols = 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6))
    fig.suptitle(get_display_title_with_kernel(local_estimator), fontsize=12)
    arrows = [(wfe.direction, wfe.energy_ratio) for wfe in local_estimator.bathymetry_estimations]

    # According to Delta_Time sign, proceed with arrow's direction inversion
    delta_time = local_estimator._bathymetry_estimations.get_estimations_attribute('delta_time')[0]
    delta_phase = local_estimator._bathymetry_estimations.get_estimations_attribute(
        'delta_phase')[0]
    corrected_arrows = []
    arrows_from_north = []
    for arrow_dir, arrow_ener in arrows:
        arrow_dir_from_north = (270 - arrow_dir)
        arrows_from_north.append((arrow_dir_from_north, arrow_ener))
        arrows = arrows_from_north
    print(' ARROW DIRECTIONS FROM NORTH =', arrows)

    if np.sign(delta_time * delta_phase) < 0:
        print('Display_polar_images_dft: inversion of arrows direction!!!!!!')
        for arrow_dir, arrow_ener in arrows:
            arrow_dir %= 180
            corrected_arrows.append((arrow_dir_from_north, arrow_ener))
            arrows = corrected_arrows

    first_image = local_estimator.ortho_sequence[0]

    # First Plot line = Image1 / pseudoRGB / Image2
    build_display_waves_image(fig, axs[0], 'Image1 [Cartesian Projection]', first_image.original_pixels,
                              resolution=first_image.resolution,
                              subplot_pos=[nrows, ncols, 1],
                              directions=arrows, cmap='gray')

    first_radon_transform = local_estimator.radon_transforms[0]
    _, directions1 = first_radon_transform.get_as_arrays()
    second_radon_transform = local_estimator.radon_transforms[1]
    sino1_fft = first_radon_transform.get_sinograms_standard_dfts()
    sino2_fft = second_radon_transform.get_sinograms_standard_dfts()

    csm_phase, spectrum_amplitude, sinograms_correlation_fft = \
        local_estimator._cross_correl_spectrum(sino1_fft, sino2_fft)
    csm_amplitude = np.abs(sinograms_correlation_fft)

    polar = csm_amplitude * csm_phase

    main_dir = local_estimator._bathymetry_estimations.get_estimations_attribute('direction')[0]
    theta_id = f'{np.int(main_dir)}'
    # Get the relevant contribution of the CSM_Ampl * CSM_Phase according to Delta_time sign
    polar *= -delta_time
    # set negative values to 0 to avoid mirror display
    polar[polar < 0] = 0
    build_polar_display(fig, axs[1], 'CSM Amplitude * CSM Phase-Shifts [Polar Projection]',
                        local_estimator, polar, first_image.resolution,
                        subplot_pos=[1, 2, 2], threshold=False)

    plt.tight_layout()
    point_id = f'{np.int(local_estimator.location.x)}_{np.int(local_estimator.location.y)}'

    plt.savefig(
        os.path.join(
            local_estimator.global_estimator._debug_path,
            "display_polar_images_debug_point_" + point_id + "_theta_" + theta_id + ".png"),
        dpi=300)
    polar_plot = plt.figure(4)
    return polar_plot


def display_radon_transforms(local_estimator: 'SpatialDFTBathyEstimator',
                             refinement_phase: bool=False) -> None:
    plt.close('all')
    fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(12, 8))
    fig.suptitle(get_display_title(local_estimator), fontsize=12)
    build_radon_transform_display(axs[:, 0], local_estimator.radon_transforms[0],
                                  'first radon transform', refinement_phase)
    build_radon_transform_display(axs[:, 2], local_estimator.radon_transforms[1],
                                  'second radon transform', refinement_phase)
    build_correl_spectrum_display(axs[:, 1], local_estimator,
                                  'Cross correlation spectrum', refinement_phase)
    plt.show()


def build_radon_transform_display(axs: Axes, transform: WavesRadon, title: str,
                                  refinement_phase: bool=False) -> None:
    values, directions = transform.get_as_arrays()
    sino_fft = transform.get_sinograms_standard_dfts()
    dft_amplitudes = np.abs(sino_fft)
    dft_phases = np.angle(sino_fft)
    variances = transform.get_sinograms_variances()
    energies = transform.get_sinograms_energies()

    build_directional_2d_display(axs[0], title, values, directions, aspect='auto', cmap='gray')
    build_directional_2d_display(axs[1], 'Sinograms DFT amplitude', dft_amplitudes, directions)
    build_directional_2d_display(axs[2], 'Sinograms DFT phase', dft_phases, directions, cmap='hsv')

    build_directional_curve_display(axs[3], 'Sinograms Variances / Energies', variances, directions)


def build_correl_spectrum_display(axs: Axes, local_estimator: 'SpatialDFTBathyEstimator',
                                  title: str, refinement_phase: bool) -> None:
    radon_transform = local_estimator.radon_transforms[0]
    if not refinement_phase:
        _, directions = radon_transform.get_as_arrays()
    else:
        directions = radon_transform.directions_interpolated_dft
    metrics = local_estimator.metrics
    key = 'interpolated_dft' if refinement_phase else 'standard_dft'
    sinograms_correlation_fft = metrics[key]['sinograms_correlation_fft']
    total_spectrum = metrics[key]['total_spectrum']
    total_spectrum_normalized = metrics[key]['total_spectrum_normalized']
    max_heta = metrics[key]['max_heta']

    build_directional_2d_display(axs[1], 'Sinograms correlation DFT module',
                                 np.abs(sinograms_correlation_fft), directions)
    build_directional_2d_display(axs[2], 'Sinograms correlation DFT Phase',
                                 np.angle(sinograms_correlation_fft), directions)
    build_directional_2d_display(axs[4], 'Sinograms correlation total spectrum',
                                 total_spectrum, directions)
    build_directional_curve_display(axs[5], 'Sinograms correlation total spectrum normalized',
                                    total_spectrum_normalized, directions)


def display_energies(local_estimator: 'SpatialDFTBathyEstimator',
                     radon1_obj: WavesRadon, radon2_obj: WavesRadon) -> None:
    fig, ax = plt.subplots()
    fig.suptitle(get_display_title(local_estimator), fontsize=12)

    image1_energy = local_estimator.ortho_sequence[0].energy_inner_disk
    image2_energy = local_estimator.ortho_sequence[1].energy_inner_disk
    ax.plot(radon1_obj.get_sinograms_energies() / image1_energy)
    ax.plot(radon2_obj.get_sinograms_energies() / image2_energy)
    plt.show()


def animate_sinograms(local_estimator: 'SpatialDFTBathyEstimator',
                      radon1_obj: WavesRadon, radon2_obj: WavesRadon) -> None:

    fig, ax = plt.subplots()
    fig.suptitle(get_display_title(local_estimator), fontsize=12)

    sinogram1_init = radon1_obj[radon1_obj.directions[0]]
    sinogram2_init = radon2_obj[radon2_obj.directions[0]]
    image1_energy = local_estimator.ortho_sequence[0].energy_inner_disk
    image2_energy = local_estimator.ortho_sequence[1].energy_inner_disk

    line1, = ax.plot(sinogram1_init.values)
    line2, = ax.plot(sinogram2_init.values)
    values1, _ = radon1_obj.get_as_arrays()
    min_radon = np.amin(values1)
    max_radon = np.amax(values1)
    plt.ylim(min_radon, max_radon)
    dir_text = ax.text(0, max_radon * 0.9, f'direction: 0, energy1: {sinogram1_init.energy}',
                       fontsize=10)

    def animate(direction: float):
        sinogram1 = radon1_obj[direction]
        sinogram2 = radon2_obj[direction]
        line1.set_ydata(sinogram1.values)  # update the data.
        line2.set_ydata(sinogram2.values)  # update the data.
        dir_text.set_text(f'direction: {direction:4.1f}, '
                          f' energy1: {sinogram1.energy/image1_energy:3.1f}, '
                          f'energy2: {sinogram2.energy/image2_energy:3.1f}')
        return line1, line2, dir_text

    ani = animation.FuncAnimation(
        fig, animate, frames=radon1_obj.directions, interval=100, blit=True, save_count=50)
    plt.show()


def display_context(local_estimator: 'SpatialDFTBathyEstimator') -> None:
    radon1 = local_estimator.radon_transforms[0]
    radon2 = local_estimator.radon_transforms[1]

    plt.close('all')
    values1, _ = radon1.get_as_arrays()
    values2, _ = radon2.get_as_arrays()
    delta_radon = np.abs(values1 - values2)
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    fig.suptitle(get_display_title(local_estimator), fontsize=12)
    axs[0, 0].imshow(radon1.pixels, aspect='auto', cmap='gray')
    axs[0, 0].set_title('subI_Det0')
    axs[1, 0].imshow(values1, aspect='auto', cmap='gray')
    axs[1, 0].set_title('radon image1')
    axs[0, 1].imshow(radon2.pixels, aspect='auto', cmap='gray')
    axs[0, 1].set_title('subI_Det1')
    axs[1, 1].imshow(values2, aspect='auto', cmap='gray')
    axs[1, 1].set_title('radon image2')
    sinograms1_energies = radon1.get_sinograms_energies()
    sinograms2_energies = radon2.get_sinograms_energies()
    image1_energy = local_estimator.ortho_sequence[0].energy_inner_disk
    image2_energy = local_estimator.ortho_sequence[1].energy_inner_disk
    axs[0, 2].plot(sinograms1_energies / image1_energy)
    axs[0, 2].plot(sinograms2_energies / image2_energy)
    axs[0, 2].set_title('directions energies')
    axs[1, 2].imshow(delta_radon, aspect='auto', cmap='gray')
    axs[1, 2].set_title('radon1 - radon2')
    plt.show()
    display_energies(local_estimator, radon1, radon2)
    animate_sinograms(local_estimator, radon1, radon2)


def floor_to_nearest_10(val):
    return np.floor(val / 10.0) * 10.0


def ceil_to_nearest_10(val):
    return np.ceil(val / 10.0) * 10.0


def sino1D_xcorr(sino1_1D, sino2_1D, correl_mode):
    length_max = max(len(sino1_1D), len(sino2_1D))
    length_min = min(len(sino1_1D), len(sino2_1D))

    if length_max == len(sino2_1D):
        lags = np.arange(-length_max + 1, length_min)
    else:
        lags = np.arange(-length_min + 1, length_max)

    cross_correl = np.correlate(
        sino1_1D / np.std(sino1_1D),
        sino2_1D / np.std(sino2_1D),
        correl_mode) / length_min
    return lags, cross_correl
