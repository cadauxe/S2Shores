# -*- coding: utf-8 -*-
"""
Class managing the computation of wave fields from two images taken at a small time interval.


:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 5 mars 2021
"""
import os
from typing import TYPE_CHECKING, List, Optional, Tuple  # @NoMove

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from bathyinversionvagues.image_processing.waves_radon import WavesRadon

if TYPE_CHECKING:
    from ..local_bathymetry.spatial_dft_bathy_estimator \
        import SpatialDFTBathyEstimator  # @UnusedImport


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
    radius = min(l1, l2) / 2
    if directions is not None:
        for direction, coeff_length in directions:
            arrow_length = radius * coeff_length
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
    ps_rgb = np.dstack((image2, image1, image2))
    ps_rgb = ps_rgb - ps_rgb.min()
    return ps_rgb / ps_rgb.max()


def build_display_pseudorgb(fig: Figure, axes: Axes, title: str, image: np.ndarray,
                            resolution: float,
                            subplot_pos: [float, float, float],
                            directions: Optional[List[Tuple[float, float]]] = None,
                            cmap: Optional[str] = None, coordinates: bool=True) -> None:

    (l1, l2, l3) = np.shape(image)
    imin = np.min(image)
    imax = np.max(image)
    #imsh = axes.imshow(image, norm=Normalize(vmin=imin, vmax=imax))
    axes.imshow(image, norm=Normalize(vmin=imin, vmax=imax))
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

    # Normalization of arrows length
    coeff_length_max = np.max((list(zip(*directions))[1]))
    radius = min(l1, l2) / 2
    if directions is not None:
        for direction, coeff_length in directions:
            arrow_length = radius * coeff_length / coeff_length_max
            dir_rad = np.deg2rad(direction)
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

    # Normalization of arrows length
    coeff_length_max = np.max((list(zip(*directions))[1]))
    radius = min(l1, l2) / 2
    if directions is not None:
        for direction, coeff_length in directions:
            arrow_length = radius * coeff_length / coeff_length_max
            dir_rad = np.deg2rad(direction)
            axes.arrow(l1 // 2, l2 // 2,
                       np.cos(dir_rad) * arrow_length, -np.sin(dir_rad) * arrow_length,
                       head_width=2, head_length=3, color='r')

    axes.xaxis.tick_top()
    axes.set_title(title, fontsize=9, loc='center')
    # Manage blank spaces
    # plt.tight_layout()


def display_waves_images_dft(local_estimator: 'SpatialDFTBathyEstimator') -> None:
    plt.close('all')
    nrows = 3
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    fig.suptitle(get_display_title_with_kernel(local_estimator), fontsize=12)
    arrows = [(wfe.direction, wfe.energy_ratio) for wfe in local_estimator.bathymetry_estimations]
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
                              subplot_pos=[nrows, ncols, 1],
                              directions=arrows, cmap='gray')
    build_display_pseudorgb(fig, axs[0, 1], 'Pseudo RGB', pseudo_rgb,
                            resolution=first_image.resolution,
                            subplot_pos=[nrows, ncols, 2],
                            directions=arrows, coordinates=False)
    build_display_waves_image(fig, axs[0, 2], 'Image2', second_image.original_pixels,
                              resolution=second_image.resolution,
                              subplot_pos=[nrows, ncols, 3],
                              directions=arrows, cmap='gray', coordinates=False)

    # Second Plot line = Image1 Filtered / pseudoRGB Filtered/ Image2 Filtered
    pseudo_rgb_filtered = create_pseudorgb(first_image.pixels, second_image.pixels)
    build_display_waves_image(fig, axs[1, 0], 'Image1 Filtered', first_image.pixels,
                              resolution=first_image.resolution,
                              subplot_pos=[nrows, ncols, 4],
                              directions=arrows, cmap='gray')
    build_display_pseudorgb(fig, axs[1, 1], 'Pseudo RGB Filtered', pseudo_rgb_filtered,
                            resolution=first_image.resolution,
                            subplot_pos=[nrows, ncols, 5],
                            directions=arrows, coordinates=False)
    build_display_waves_image(fig, axs[1, 2], 'Image2 Filtered', second_image.pixels,
                              resolution=second_image.resolution,
                              subplot_pos=[nrows, ncols, 6],
                              directions=arrows, cmap='gray', coordinates=False)

    # Third Plot line = Image1 Circle Filtered / pseudoRGB Circle Filtered/ Image2 Circle Filtered
    image1_circle_filtered = first_image.pixels * first_image.circle_image
    image2_circle_filtered = second_image.pixels * second_image.circle_image
    pseudo_rgb_circle_filtered = create_pseudorgb(image1_circle_filtered, image2_circle_filtered)
    build_display_waves_image(fig, axs[2, 0], 'Image1 Circle Filtered', image1_circle_filtered,
                              resolution=first_image.resolution,
                              subplot_pos=[nrows, ncols, 7],
                              directions=arrows, cmap='gray')
    build_display_pseudorgb(fig, axs[2, 1], 'Pseudo RGB Circle Filtered', pseudo_rgb_circle_filtered,
                            resolution=first_image.resolution,
                            subplot_pos=[nrows, ncols, 8],
                            directions=arrows, coordinates=False)
    build_display_waves_image(fig, axs[2, 2], 'Image2 Circle Filtered', image2_circle_filtered,
                              resolution=second_image.resolution,
                              subplot_pos=[nrows, ncols, 9],
                              directions=arrows, cmap='gray', coordinates=False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            local_estimator.global_estimator._debug_path,
            "display_waves_images.png"),
        dpi=300)
    plt.show()


def build_sinogram_display(axes: Axes, title: str, values1: np.ndarray, directions: np.ndarray,
                           values2: np.ndarray,
                           ordonate: bool=True, **kwargs: dict) -> None:
    #extent = [np.min(directions), np.max(directions), 0, values1.shape[0]]
    #imin = np.min(values1)
    #imax = np.max(values1)
    #axes.imshow(values1, norm=Normalize(vmin=imin, vmax=imax), extent=extent, **kwargs)
    extent = [np.min(directions), np.max(directions),
              np.ceil(-values1.shape[0] / 2),
              np.floor(values1.shape[0] / 2)]
    axes.imshow(values1, aspect='auto', extent=extent, **kwargs)
    axes.plot(directions,
              (np.var(values2, axis=0) / np.max(np.var(values2, axis=0)) - 0.5) * values2.shape[0],
              color="red", lw=1, ls='--')
    axes.plot(directions,
              (np.var(values1, axis=0) / np.max(np.var(values1, axis=0)) - 0.5) * values1.shape[0],
              color="white", lw=0.8)

    axes.grid(lw=0.5, color='white', alpha=0.7, linestyle='-')
    axes.set_xlim(-120, 120)
    axes.set_xticks(np.arange(-120, 121, 40))
    if ordonate:
        axes.set_ylabel(r'$\rho$ [pixels]', fontsize=8)
    else:
        axes.yaxis.set_ticklabels([])
    plt.setp(axes.get_xticklabels(), fontsize=8)
    axes.set_title(title, fontsize=10)
    axes.tick_params(axis='both', which='major', labelsize=8)


def build_sinogram_difference_display(axes: Axes, title: str, values: np.ndarray,
                                      directions: np.ndarray, cmap: Optional[str] = None,
                                      **kwargs: dict) -> None:
    #extent = [np.min(directions), np.max(directions), 0, values.shape[0]]
    #imin = np.min(values)
    #imax = np.max(values)
    #axes.imshow(values, norm=Normalize(vmin=imin, vmax=imax), cmap=cmap, extent=extent, **kwargs)
    extent = [np.min(directions), np.max(directions),
              np.ceil(-values.shape[0] / 2),
              np.floor(values.shape[0] / 2)]
    axes.imshow(values, cmap=cmap, aspect='auto', extent=extent, **kwargs)
    axes.grid(lw=0.5, color='black', alpha=0.7, linestyle='-')
    axes.yaxis.set_ticklabels([])
    axes.set_xlim(-120, 120)
    axes.set_xticks(np.arange(-120, 121, 40))
    plt.setp(axes.get_xticklabels(), fontsize=8)
    axes.set_title(title, fontsize=10)
    axes.tick_params(axis='both', which='major', labelsize=8)


def display_dft_sinograms(local_estimator: 'SpatialDFTBathyEstimator') -> None:
    plt.close('all')
    nrows = 2
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
    fig.suptitle(get_display_title_with_kernel(local_estimator), fontsize=12)
    arrows = [(wfe.direction, wfe.energy_ratio) for wfe in local_estimator.bathymetry_estimations]
    first_image = local_estimator.ortho_sequence[0]
    second_image = local_estimator.ortho_sequence[1]

    # First Plot line = Image1 Circle Filtered / pseudoRGB Circle Filtered/ Image2 Circle Filtered
    image1_circle_filtered = first_image.pixels * first_image.circle_image
    image2_circle_filtered = second_image.pixels * second_image.circle_image
    pseudo_rgb_circle_filtered = create_pseudorgb(image1_circle_filtered, image2_circle_filtered)
    build_display_waves_image(fig, axs[0, 0], 'Image1 Circle Filtered', image1_circle_filtered,
                              subplot_pos=[nrows, ncols, 1],
                              resolution=first_image.resolution,
                              directions=arrows, cmap='gray')
    build_display_pseudorgb(fig, axs[0, 1], 'Pseudo RGB Circle Filtered', pseudo_rgb_circle_filtered,
                            resolution=first_image.resolution,
                            subplot_pos=[nrows, ncols, 2],
                            directions=arrows, coordinates=False)
    build_display_waves_image(fig, axs[0, 2], 'Image2 Circle Filtered', image2_circle_filtered,
                              resolution=second_image.resolution,
                              subplot_pos=[nrows, ncols, 3],
                              directions=arrows, cmap='gray', coordinates=False)

    # Second Plot line = Sinogram1 / Sinogram2-Sinogram1 / Sinogram2
    first_radon_transform = local_estimator.radon_transforms[0]
    sinogram1, directions1 = first_radon_transform.get_as_arrays()
    second_radon_transform = local_estimator.radon_transforms[1]
    sinogram2, directions2 = second_radon_transform.get_as_arrays()
    radon_difference = np.abs(sinogram2 - sinogram1)

    build_sinogram_display(
        axs[1, 0], 'Sinogram1 [Radon Transform on Image1]', sinogram1, directions1, sinogram2)
    build_sinogram_difference_display(
        axs[1, 1], 'Sinogram2 - Sinogram1', radon_difference, directions2, cmap='bwr')
    build_sinogram_display(
        axs[1, 2], 'Sinogram2 [Radon Transform on Image2]', sinogram2, directions2, sinogram1,
        ordonate=False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            local_estimator.global_estimator._debug_path,
            "display_sinograms.png"),
        dpi=300)
    plt.show()


def build_sinogram_spectral_display(axes: Axes, title: str, values: np.ndarray,
                                    directions: np.ndarray, kfft: np.ndarray,
                                    ordonate: bool=True, **kwargs: dict) -> None:
    extent = [np.min(directions), np.max(directions), 0, kfft.max()]
    axes.imshow(values, aspect='auto', origin="lower", extent=extent, **kwargs)
    axes.plot(directions, ((np.var(values, axis=0) / np.max(np.var(values, axis=0))) * kfft.max()),
              color="white", lw=0.7)
    axes.grid(lw=0.5, color='white', alpha=0.7, linestyle='-')
    axes.set_xlim(-120, 120)
    axes.set_xticks(np.arange(-120, 121, 40))

    if ordonate:
        axes.set_ylabel(r'Wavenumber $\nu$ [m$^{-1}$]', fontsize=8)
    else:
        axes.yaxis.set_ticklabels([])
    plt.setp(axes.get_xticklabels(), fontsize=8)
    axes.set_title(title, fontsize=10)
    axes.tick_params(axis='both', which='major', labelsize=8)


def build_sinogram_fft_display(axes: Axes, title: str, values: np.ndarray, directions: np.ndarray,
                               kfft: np.ndarray, type: str, ordonate: bool=True,
                               **kwargs: dict) -> None:

    extent = [np.min(directions), np.max(directions), 0, kfft.max()]
    axes.imshow(values, aspect='auto', origin="lower", extent=extent, **kwargs)
    if type == 'amplitude':
        axes.plot(directions, ((np.var(values, axis=0) / np.max(np.var(values, axis=0))) * kfft.max()),
                  color="white", lw=0.7)
        axes.plot(directions, ((np.max(values, axis=0) / np.max(np.max(values, axis=0))) * kfft.max()),
                  color="orange", lw=0.7)

    axes.grid(lw=0.5, color='white', alpha=0.7, linestyle='-')
    axes.set_xlim(-120, 120)
    axes.set_xticks(np.arange(-120, 121, 40))
    if ordonate:
        axes.set_ylabel(r'Wavenumber $\nu$ [m$^{-1}$]', fontsize=8)
    else:
        axes.yaxis.set_ticklabels([])
    plt.setp(axes.get_xticklabels(), fontsize=8)
    axes.set_title(title, fontsize=10)
    axes.tick_params(axis='both', which='major', labelsize=8)


def build_correl_spectrum_matrix(axes: Axes, local_estimator: 'SpatialDFTBathyEstimator',
                                 sino1_fft: np.ndarray, sino2_fft: np.ndarray, kfft: np.ndarray,
                                 type: str, title: str, ordonate: bool=True,
                                 refinement_phase: bool=False) -> None:
    radon_transform = local_estimator.radon_transforms[0]
    if not refinement_phase:
        _, directions = radon_transform.get_as_arrays()
    else:
        directions = radon_transform.directions_interpolated_dft
    metrics = local_estimator.metrics
    key = 'interpolated_dft' if refinement_phase else 'standard_dft'
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
        build_sinogram_fft_display(axes, title, csm_amplitude, directions, kfft, type, ordonate)
    if type == 'phase':
        build_sinogram_fft_display(axes, title, csm_phase, directions, kfft, type, ordonate)


def display_dft_sinograms_spectral_analysis(local_estimator: 'SpatialDFTBathyEstimator') -> None:
    plt.close('all')
    nrows = 3
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
    fig.suptitle(get_display_title_with_kernel(local_estimator), fontsize=12)

    # First Plot line = Sinogram1 / Sinogram2-Sinogram1 / Sinogram2
    first_radon_transform = local_estimator.radon_transforms[0]
    sinogram1, directions1 = first_radon_transform.get_as_arrays()
    second_radon_transform = local_estimator.radon_transforms[1]
    sinogram2, directions2 = second_radon_transform.get_as_arrays()
    radon_difference = np.abs(sinogram2 - sinogram1)

    build_sinogram_display(
        axs[0, 0], 'Sinogram1 [Radon Transform on Image1]', sinogram1, directions1, sinogram2)
    build_sinogram_difference_display(
        axs[0, 1], 'Sinogram2 - Sinogram1', radon_difference, directions2, cmap='bwr')
    build_sinogram_display(
        axs[0, 2], 'Sinogram2 [Radon Transform on Image2]', sinogram2, directions2, sinogram1,
        ordonate=False)

    # Second Plot line = Spectral Amplitude Sinogram 1 / CSM Amplitude /
    # Spectral Amplitude Sinogram 2

    sino1_fft = first_radon_transform.get_sinograms_standard_dfts()
    sino2_fft = second_radon_transform.get_sinograms_standard_dfts()
    kfft = local_estimator._metrics['kfft']

    build_sinogram_spectral_display(
        axs[1, 0], 'Spectral Amplitude Sinogram1 DFT',
        np.abs(sino1_fft), directions1, kfft)
    build_correl_spectrum_matrix(
        axs[1, 1], local_estimator, sino1_fft, sino2_fft, kfft, 'amplitude',
        'Cross Spectral Matrix (Amplitude)', ordonate=False)
    build_sinogram_spectral_display(
        axs[1, 2], 'Spectral Amplitude Sinogram2 DFT',
        np.abs(sino2_fft), directions2, kfft, ordonate=False)

    # Third Plot line = Spectral Amplitude  * CSM Phase Sinogram 1 / CSM Phase /
    # Spectral Amplitude * CSM Phase Sinogram 2
    # Manage blank spaces"
    csm_phase, spectrum_amplitude, sinograms_correlation_fft = \
        local_estimator._cross_correl_spectrum(sino1_fft, sino2_fft)
    build_sinogram_spectral_display(
        axs[2, 0], 'Spectral Amplitude * CSM_Phase Sinogram1 DFT',
        np.abs(sino1_fft) * csm_phase, directions1, kfft)
    build_correl_spectrum_matrix(
        axs[2, 1], local_estimator, sino1_fft, sino2_fft, kfft, 'phase',
        'Cross Spectral Matrix (Phase-shifts)', ordonate=False)
    build_sinogram_spectral_display(
        axs[2, 2], 'Spectral Amplitude * CSM_Phase Sinogram2 DFT',
        np.abs(sino2_fft) * csm_phase, directions2, kfft, ordonate=False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            local_estimator.global_estimator._debug_path,
            "display_sinograms_spectral_analysis.png"),
        dpi=300)
    plt.show()


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
