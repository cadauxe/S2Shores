# -*- coding: utf-8 -*-
"""
Functions for displaying sinograms and related visualizations.

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2024 CNES. All rights reserved.
:created: 4 November 2024
:license: see LICENSE file


  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
import sys
import pprint

pprint.pprint(sys.path)

import sys
import pprint
import os
from typing import Optional

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.ticker as mticker
from s2shores.generic_utils.image_utils import normalized_cross_correlation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from s2shores.image_processing.waves_radon import WavesRadon

from .wave_images_display import build_display_waves_image
from .pseudorgb_display import build_display_pseudorgb, create_pseudorgb
from .display_utils import (floor_to_nearest_10,
                            ceil_to_nearest_10, get_display_title,
                            get_display_title_with_kernel)


def build_sinogram_display(axes: Axes, title: str, values1: np.ndarray, directions: np.ndarray,
                           values2: np.ndarray, main_theta: float, plt_min: float, plt_max: float,
                           ordonate: bool=True, abscissa: bool=True, master: bool=True,
                           **kwargs: dict) -> None:
    """Build a sinogram display with given parameters."""
    extent = [np.min(directions), np.max(directions),
              np.floor(-values1.shape[0] / 2),
              np.ceil(values1.shape[0] / 2)]
    axes.imshow(values1, aspect='auto', extent=extent, **kwargs)
    normalized_var1 = (np.var(values1, axis=0) /
                       np.max(np.var(values1, axis=0)) - 0.5) * values1.shape[0]
    normalized_var2 = (np.var(values2, axis=0) /
                       np.max(np.var(values2, axis=0)) - 0.5) * values2.shape[0]
    axes.plot(directions, normalized_var2,
              color='red', lw=1, ls='--', label='Normalized Variance \n Comparative Sinogram')
    axes.plot(directions, normalized_var1,
              color='white', lw=0.8, label='Normalized Variance \n Reference Sinogram')

    pos1 = np.where(normalized_var1 == np.max(normalized_var1))
    max_var_theta = directions[pos1][0]
    # Check coherence of main direction between Master / Slave
    if max_var_theta * main_theta < 0:
        max_var_theta = max_var_theta % (np.sign(main_theta) * 180.0)
    # Check if the direction belongs to the plotting interval [plt_min:plt_max]
    if max_var_theta < plt_min or max_var_theta > plt_max:
        max_var_theta %= -np.sign(max_var_theta) * 180.0
    theta_label = '$\Theta${:.1f}° [Variance Max]'.format(max_var_theta)
    theta_label_orig = '$\Theta${:.1f}° [Main Direction]'.format(main_theta)

    axes.axvline(max_var_theta, np.floor(-values1.shape[0] / 2), np.ceil(values1.shape[0] / 2),
                 color='orange', ls='--', lw=1, label=theta_label)

    axes.axvline(main_theta, np.floor(-values1.shape[0] / 2), np.ceil(values1.shape[0] / 2),
                 color='blue', ls='--', lw=1, label=theta_label_orig)

    legend = axes.legend(loc='upper right', shadow=True, fontsize=6)
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('C0')
    axes.grid(lw=0.5, color='white', alpha=0.7, linestyle='-')
    axes.set_xlim(plt_min, plt_max)
    axes.set_xticks(np.arange(plt_min, plt_max + 1, 45))
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
                                      directions: np.ndarray, plt_min: float, plt_max: float,
                                      abscissa: bool=True, cmap: Optional[str] = None,
                                      **kwargs: dict) -> None:

    extent = [np.min(directions), np.max(directions),
              np.floor(-values.shape[0] / 2),
              np.ceil(values.shape[0] / 2)]

    axes.imshow(values, cmap=cmap, aspect='auto', extent=extent, **kwargs)

    axes.grid(lw=0.5, color='black', alpha=0.7, linestyle='-')
    axes.yaxis.set_ticklabels([])
    axes.set_xlim(plt_min, plt_max)
    axes.set_xticks(np.arange(plt_min, plt_max + 1, 45))
    plt.setp(axes.get_xticklabels(), fontsize=8)

    if abscissa:
        axes.set_xlabel(r'Direction Angle $\theta$ [degrees]', fontsize=8)

    axes.set_title(title, fontsize=10)
    axes.tick_params(axis='both', which='major', labelsize=8)


def display_dft_sinograms(local_estimator: 'SpatialDFTBathyEstimator') -> None:
    fig, axs, first_image, second_image = get_parameters_for_sinogram_display(local_estimator)
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


    point_id = manage_radon_difference(radon_difference,
                                       sinogram1, directions1,
                                       sinogram2, directions2,
                                       axs, main_direction, local_estimator)

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


def get_parameters_for_sinogram_display(
        local_estimator: 'SpatialCorrelationBathyEstimator',
        directions=False) -> None:
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

    if directions:
        # Since wfe.energy_ratio not available for SpatialCorrelation:
        default_arrow_length = np.shape(first_image.original_pixels)[0]
        arrows = [(wfe.direction, default_arrow_length)
              for wfe in local_estimator.bathymetry_estimations]

        build_display_waves_image(fig, axs[0, 0], 'Master Image Circle Filtered',
                                  image1_circle_filtered, subplot_pos=[nrows, ncols, 1],
                                  resolution=first_image.resolution, directions=arrows,
                                  cmap='gray')
    else:
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

    if directions:
        build_display_waves_image(fig,axs[0, 2],
                            'Slave Image Circle Filtered',
                            image2_circle_filtered, resolution=second_image.resolution,
                            subplot_pos=[nrows, ncols, 3],
                            directions=arrows, cmap='gray', coordinates=False)
    else:
        build_display_waves_image(fig, axs[0, 2],
                            'Image2 Circle Filtered',
                            image2_circle_filtered,
                            resolution=second_image.resolution,
                            subplot_pos=[nrows, ncols, 3], cmap='gray', coordinates=False)


    return fig, axs, first_image, second_image

def manage_radon_difference(radon_difference, sinogram1, directions1,
                            sinogram2, directions2, axs: Axes, main_direction,
        local_estimator: 'SpatialCorrelationBathyEstimator') -> int:
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
    return point_id


def display_sinograms_spatial_correlation(
        local_estimator: 'SpatialCorrelationBathyEstimator') -> None:
    fig, axs, first_image, second_image = get_parameters_for_sinogram_display(local_estimator,
                                                                            directions=True)


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

    point_id = manage_radon_difference(radon_difference,
                                       sinogram1, directions1,
                                       sinogram2, directions2,
                                       axs, main_direction, local_estimator)

    theta_id = f'{int(main_direction)}'
    plt.savefig(
        os.path.join(
            local_estimator.global_estimator.debug_path,
            'display_sinograms_debug_point_' + point_id + '_theta_' + theta_id + '.png'),
        dpi=300)
    # plt.show()
    dft_sino = plt.figure(2)
    return dft_sino


def build_sinogram_spectral_display(
        axes: Axes,
        title: str,
        values: np.ndarray,
        directions: np.ndarray,
        kfft: np.ndarray,
        plt_min: float,
        plt_max: float,
        ordonate: bool=True,
        abscissa: bool=True,
        **kwargs: dict) -> None:
    extent = [np.min(directions), np.max(directions), 0.0, kfft.max()]
    im = axes.imshow(values, aspect='auto', origin='lower', extent=extent, **kwargs)

    axes.plot(directions, ((np.max(values, axis=0) / np.max(np.max(values, axis=0))) * kfft.max()),
              color='black', lw=0.7, label='Normalized Maximum')

    # colorbar
    cbbox = inset_axes(axes, '50%', '10%', loc='upper left')
    [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
    cbbox.tick_params(
        axis='both',
        left=False,
        top=False,
        right=False,
        bottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
        labelbottom=False)
    cbbox.set_facecolor([1, 1, 1, 0.7])
    cbaxes = inset_axes(cbbox, '70%', '20%', loc='upper center')

    cbar = plt.colorbar(
        im,
        cax=cbaxes,
        ticks=[
            np.nanmin(values),
            np.nanmax(values)],
        orientation='horizontal')
    cbar.ax.tick_params(labelsize=5)
    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cbar.ax.xaxis.set_major_formatter(f)
    cbar.ax.xaxis.get_offset_text().set_fontsize(4)

    legend = axes.legend(loc='upper right', shadow=True, fontsize=6)
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('C0')

    axes.grid(lw=0.5, color='white', alpha=0.7, linestyle='-')
    axes.set_xlim(plt_min, plt_max)
    axes.set_xticks(np.arange(plt_min, plt_max + 1, 45))
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
                               kfft: np.ndarray, plt_min: float, plt_max: float, type: str,
                               ordonate: bool=True, abscissa: bool=True, **kwargs: dict) -> None:

    extent = [np.min(directions), np.max(directions), 0.0, kfft.max()]
    im = axes.imshow(values, aspect='auto', origin='lower', extent=extent, **kwargs)

    # colorbar
    cbbox = inset_axes(axes, '50%', '10%', loc='upper left')
    [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
    cbbox.tick_params(
        axis='both',
        left=False,
        top=False,
        right=False,
        bottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
        labelbottom=False)
    cbbox.set_facecolor([1, 1, 1, 0.7])
    cbaxes = inset_axes(cbbox, '70%', '20%', loc='upper center')

    cbar = plt.colorbar(
        im,
        cax=cbaxes,
        ticks=[
            np.nanmin(values),
            np.nanmax(values)],
        orientation='horizontal')
    cbar.ax.tick_params(labelsize=5)
    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cbar.ax.xaxis.set_major_formatter(f)
    cbar.ax.xaxis.get_offset_text().set_fontsize(4)

    if type == 'amplitude':
        axes.plot(directions, ((np.var(values, axis=0) / np.max(np.var(values, axis=0)))
                  * kfft.max()), color='white', lw=0.7, label='Normalized Variance')
        axes.plot(directions, ((np.max(values, axis=0) / np.max(np.max(values, axis=0)))
                  * kfft.max()), color='orange', lw=0.7, label='Normalized Maximum')
        legend = axes.legend(loc='upper right', shadow=True, fontsize=6)
        # Put a nicer background color on the legend.
        legend.get_frame().set_facecolor('C0')
    axes.grid(lw=0.5, color='white', alpha=0.7, linestyle='-')
    axes.set_xlim(plt_min, plt_max)
    axes.set_xticks(np.arange(plt_min, plt_max + 1, 45))
    plt.setp(axes.get_xticklabels(), fontsize=8)

    if ordonate:
        axes.set_ylabel(r'Wavenumber $\nu$ [m$^{-1}$]', fontsize=8)
    else:
        axes.yaxis.set_ticklabels([])
    if abscissa:
        axes.set_xlabel(r'Direction Angle $\theta$ [degrees]', fontsize=8)

    axes.set_title(title, fontsize=10)
    axes.tick_params(axis='both', which='major', labelsize=8)


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

def build_sinogram_1D_display_master(
        axes: Axes,
        title: str,
        values1: np.ndarray,
        directions: np.ndarray,
        main_theta: float,
        plt_min: float,
        plt_max: float,
        ordonate: bool=True,
        abscissa: bool=True,
        **kwargs: dict) -> None:

  #  index_theta = int(main_theta - np.min(directions))
    index_theta = np.where(directions == int(main_theta))[0]

    # Check if the main direction belongs to the plotting interval [plt_min:plt_max]
    if main_theta < plt_min or main_theta > plt_max:
        main_theta %= -np.sign(main_theta) * 180.0
    theta_label = 'Sinogram 1D along \n$\Theta$={:.1f}°'.format(main_theta)
    nb_pixels = np.shape(values1[:, index_theta])[0]
    absc = np.arange(-nb_pixels / 2, nb_pixels / 2)
    axes.plot(absc, np.flip((values1[:, index_theta] / np.max(np.abs(values1[:, index_theta])))),
              color='orange', lw=0.8, label=theta_label)

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


def build_sinogram_1D_display_slave(
        axes: Axes,
        title: str,
        values: np.ndarray,
        directions: np.ndarray,
        main_theta: float,
        plt_min: float,
        plt_max: float,
        ordonate: bool=True,
        abscissa: bool=True,
        **kwargs: dict) -> None:

    normalized_var = (np.var(values, axis=0) /
                      np.max(np.var(values, axis=0)) - 0.5) * values.shape[0]
    pos = np.where(normalized_var == np.max(normalized_var))
    main_theta_slave = directions[pos][0]

    # Check coherence of main direction between Master / Slave
    if directions[pos][0] * main_theta < 0:
        main_theta_slave = directions[pos][0] % (np.sign(main_theta) * 180.0)

    index_theta_master = np.where(directions == int(main_theta))[0]
    index_theta_slave = pos[0]


    # Check if the main direction belongs to the plotting interval [plt_min:plt_max]
    if main_theta < plt_min or main_theta > plt_max:
        main_theta %= -np.sign(main_theta) * 180.0
    if main_theta_slave < plt_min or main_theta_slave > plt_max:
        main_theta_slave %= -np.sign(main_theta_slave) * 180.0
    theta_label_master = 'along Master Main Direction\n$\Theta$={:.1f}°'.format(main_theta)
    theta_label_slave = 'along Slave Main Direction\n$\Theta$={:.1f}°'.format(main_theta_slave)
    nb_pixels = np.shape(values[:, index_theta_master])[0]
    absc = np.arange(-nb_pixels / 2, nb_pixels / 2)
    axes.plot(absc,
              np.flip((values[:,
                              index_theta_master] / np.max(np.abs(values[:,
                                                                         index_theta_master])))),
              color='orange',
              lw=0.8,
              label=theta_label_master)
    axes.plot(absc,
              np.flip((values[:,
                              index_theta_slave] / np.max(np.abs(values[:,
                                                                        index_theta_slave])))),
              color='blue',
              lw=0.8,
              ls='--',
              label=theta_label_slave)

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


def build_sinogram_1D_cross_correlation(
        axes: Axes,
        title: str,
        values1: np.ndarray,
        directions1: np.ndarray,
        main_theta: float,
        values2: np.ndarray,
        directions2: np.ndarray,
        plt_min: float,
        plt_max: float,
        correl_mode: str,
        ordonate: bool=True,
        abscissa: bool=True,
        **kwargs: dict) -> None:

    normalized_var = (np.var(values2, axis=0) /
                      np.max(np.var(values2, axis=0)) - 0.5) * values2.shape[0]
    pos2 = np.where(normalized_var == np.max(normalized_var))
    main_theta_slave = directions2[pos2][0]
    # Check coherence of main direction between Master / Slave
    if directions2[pos2][0] * main_theta < 0:
        main_theta_slave = directions2[pos2][0] % (np.sign(main_theta) * 180.0)

    index_theta1 = int(np.where(directions1 == int(main_theta))[0])
    # get 1D-sinogram1 along relevant direction
    sino1_1D = values1[:, index_theta1]
    # theta_label1 = 'Sinogram1 1D'  # along \n$\Theta$={:.1f}°'.format(main_theta)
    nb_pixels1 = np.shape(values1[:, index_theta1])[0]
    absc = np.arange(-nb_pixels1 / 2, nb_pixels1 / 2)
    # axes.plot(absc, np.flip((values1[:, index_theta1] / np.max(np.abs(values1[:, index_theta1])))),
    #          color='orange', lw=0.8, label=theta_label1)

    index_theta2_master = int(np.where(directions2 == int(main_theta))[0])
    index_theta2_slave = int(pos2[0][0])

    # get 1D-sinogram2 along relevant direction
    sino2_1D_master = values2[:, index_theta2_master]
    # theta_label2_master = 'Sinogram2 1D MASTER'  # along \n$\Theta$={:.1f}°'.format(main_theta)
    #nb_pixels2 = np.shape(values2[:, index_theta2_master])[0]
    #absc2 = np.arange(-nb_pixels2 / 2, nb_pixels2 / 2)
    # axes.plot(absc2, np.flip((values2[:, index_theta2_master] / np.max(np.abs(values2[:, index_theta2_master])))),
    #          color='black', lw=0.8, ls='--', label=theta_label2_master)

    # Check if the main direction belongs to the plotting interval [plt_min:plt_max]
    if main_theta < plt_min or main_theta > plt_max:
        main_theta_label = main_theta % (-np.sign(main_theta) * 180.0)
    else:
        main_theta_label = main_theta
    if main_theta_slave < plt_min or main_theta_slave > plt_max:
        main_theta_slave_label = main_theta_slave % (-np.sign(main_theta_slave) * 180.0)
    else:
        main_theta_slave_label = main_theta_slave

    # Compute Cross-Correlation between Sino1 [Master Man Direction] & Sino2 [Master Main Direction]
    sino_cross_corr_norm_master = normalized_cross_correlation(
        np.flip(sino1_1D), np.flip(sino2_1D_master), correl_mode)
    label_correl_master = 'Sino1_1D[$\Theta$={:.1f}°] vs Sino2_1D[$\Theta$={:.1f}°]'.format(
        main_theta_label, main_theta_label)
    axes.plot(absc, sino_cross_corr_norm_master, color='red', lw=0.8, label=label_correl_master)

    sino2_1D_slave = values2[:, index_theta2_slave]
    # theta_label2_slave = 'Sinogram2 1D SLAVE'  # along \n$\Theta$={:.1f}°'.format(main_theta)
    # axes.plot(absc2, np.flip((values2[:, index_theta2_slave] / np.max(np.abs(values2[:, index_theta2_slave])))),
    #          color='green', lw=0.8, ls='--', label=theta_label2_slave)
    # Compute Cross-Correlation between Sino1 [Master Main Direction& Sino2 [Slave Main Direction]
    sino_cross_corr_norm_slave = normalized_cross_correlation(
        np.flip(sino1_1D), np.flip(sino2_1D_slave), correl_mode)

    label_correl_slave = 'Sino1_1D[$\Theta$={:.1f}°] vs Sino2_1D[$\Theta$={:.1f}°]'.format(
        main_theta_label, main_theta_slave_label)
    axes.plot(absc, sino_cross_corr_norm_slave, color='black', ls='--', lw=0.8,
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


def build_sinogram_2D_cross_correlation(
        axes: Axes,
        title: str,
        values1: np.ndarray,
        directions1: np.ndarray,
        main_theta: float,
        values2: np.ndarray,
        plt_min: float,
        plt_max: float,
        correl_mode: str,
        choice: str,
        imgtype: str,
        ordonate: bool=True,
        abscissa: bool=True,
        cmap: Optional[str] = None,
        **kwargs: dict) -> None:

    extent = [np.min(directions1), np.max(directions1),
              np.floor(-values1.shape[0] / 2),
              np.ceil(values1.shape[0] / 2)]

    if imgtype == 'slave':
        normalized_var = (np.var(values1, axis=0) /
                          np.max(np.var(values1, axis=0)) - 0.5) * values1.shape[0]
        pos = np.where(normalized_var == np.max(normalized_var))
        slave_main_theta = directions1[pos][0]
        # Check coherence of main direction between Master / Slave
        if slave_main_theta * main_theta < 0:
            slave_main_theta = slave_main_theta % (np.sign(main_theta) * 180.0)
        # Check if the main direction belongs to the plotting interval [plt_min:plt_max]
        if slave_main_theta < plt_min or slave_main_theta > plt_max:
            slave_main_theta = slave_main_theta % (-np.sign(slave_main_theta) * 180.0)

        main_theta = slave_main_theta
        title = 'Normalized Cross-Correlation Signal between \n Sino2[$\Theta$={:.1f}°] and Sino1[All Directions]'.format(
            main_theta)

    if choice == 'one_dir':
        index_theta1 = int(np.where(directions1 == int(main_theta))[0])
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
        normalized_var_val3 = (np.var(np.transpose(values3),
                                      axis=0) / np.max(np.var(np.transpose(values3),
                                                              axis=0)) - 0.5) * np.transpose(values3).shape[0]

        axes.plot(directions1, normalized_var_val3,
                  color='white', lw=1, ls='--', label='Normalized Variance', zorder=5)

        # Find position of the local maximum of the normalized variance of values3
        pos_val3 = np.where(normalized_var_val3 == np.max(normalized_var_val3))
        max_var_pos = directions1[pos_val3][0]

        # Check coherence of main direction between Master / Slave
        if directions1[pos_val3][0] * main_theta < 0:
            max_var_pos = directions1[pos_val3][0] % (np.sign(main_theta) * 180.0)
        # Check if the main direction belongs to the plotting interval [plt_min:plt_max]
        if max_var_pos < plt_min or max_var_pos > plt_max:
            max_var_pos %= -np.sign(max_var_pos) * 180.0

        max_var_label = '$\Theta$={:.1f}° [Variance Max]'.format(max_var_pos)
        axes.axvline(max_var_pos, np.floor(-values1.shape[0] / 2), np.ceil(values1.shape[0] / 2),
                     color='red', ls='--', lw=1, label=max_var_label, zorder=10)

    # Main 2D-plot
    axes.imshow(np.transpose(values3), cmap=cmap, aspect='auto', extent=extent, **kwargs)

    # Check if the main direction belongs to the plotting interval [plt_min:plt_max]
    if main_theta < plt_min or main_theta > plt_max:
        main_theta %= -np.sign(main_theta) * 180.0
    theta_label = '$\Theta$={:.1f}°'.format(main_theta)
    axes.axvline(main_theta, np.floor(-values1.shape[0] / 2), np.ceil(values1.shape[0] / 2),
                 color='orange', ls='--', lw=1, label=theta_label)

    legend = axes.legend(loc='upper right', shadow=True, fontsize=6)
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('C0')

    axes.grid(lw=0.5, color='black', alpha=0.7, linestyle='-')
    axes.set_xlim(plt_min, plt_max)
    axes.set_xticks(np.arange(plt_min, plt_max + 1, 45))
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


def build_correl_spectrum_matrix(axes: Axes, local_estimator: 'SpatialDFTBathyEstimator',
                                 sino1_fft: np.ndarray, sino2_fft: np.ndarray, kfft: np.ndarray,
                                 plt_min: float, plt_max: float, type: str, title: str,
                                 refinement_phase: bool=False) -> None:
    radon_transform = local_estimator.radon_transforms[0]
    if not refinement_phase:
        _, directions = radon_transform.get_as_arrays()
    else:
        directions = radon_transform.directions_interpolated_dft

    csm_phase, spectrum_amplitude, sinograms_correlation_fft = \
        local_estimator._cross_correl_spectrum(sino1_fft, sino2_fft)

    csm_amplitude = np.abs(sinograms_correlation_fft)

    if type == 'amplitude':
        build_sinogram_fft_display(axes, title, csm_amplitude, directions, kfft, plt_min, plt_max,
                                   type, ordonate=False, abscissa=False)
    if type == 'phase':
        build_sinogram_fft_display(
            axes,
            title,
            csm_amplitude *
            csm_phase,
            directions,
            kfft,
            plt_min,
            plt_max,
            type,
            ordonate=False)