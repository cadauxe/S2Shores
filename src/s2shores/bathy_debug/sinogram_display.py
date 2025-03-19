# -*- coding: utf-8 -*-
"""
Class managing the different displays based on sinograms.


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

from typing import TYPE_CHECKING, Optional  # @NoMove

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..generic_utils.image_utils import normalized_cross_correlation
from .display_utils import ceil_to_nearest_10, floor_to_nearest_10

if TYPE_CHECKING:
    from ..local_bathymetry.spatial_dft_bathy_estimator import (
        SpatialDFTBathyEstimator)  # @UnusedImport


def build_sinogram_display(axes: Axes, title: str, values1: np.ndarray, directions: np.ndarray,
                           values2: np.ndarray, main_theta: float, plt_min: float, plt_max: float,
                           ordonate: bool = True, abscissa: bool = True, master: bool = True,
                           **kwargs: dict) -> None:
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
        ordonate: bool = True,
        abscissa: bool = True,
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
    # nb_pixels2 = np.shape(values2[:, index_theta2_master])[0]
    # absc2 = np.arange(-nb_pixels2 / 2, nb_pixels2 / 2)
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
        ordonate: bool = True,
        abscissa: bool = True,
        cmap: Optional[str] = None,
        **kwargs: dict) -> None:
    extent = [np.min(directions1), np.max(directions1),
              np.floor(-values1.shape[0] / 2),
              np.ceil(values1.shape[0] / 2)]

    if imgtype == 'slave':
        main_theta, title = slave_sinogram_2D_cross_corr(directions1, main_theta, plt_max, plt_min, title, values1)

    if choice == 'one_dir':
        pos_max, values3 = one_dir_sinogram_2D_cross_corr(correl_mode, directions1, main_theta, values1, values2)

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


def slave_sinogram_2D_cross_corr(directions1, main_theta, plt_max, plt_min, title, values1):
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
    return main_theta, title


def one_dir_sinogram_2D_cross_corr(correl_mode, directions1, main_theta, values1, values2):
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
    return pos_max, values3


def build_sinogram_spectral_display(
        axes: Axes,
        title: str,
        values: np.ndarray,
        directions: np.ndarray,
        kfft: np.ndarray,
        plt_min: float,
        plt_max: float,
        ordonate: bool = True,
        abscissa: bool = True,
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


def build_correl_spectrum_matrix(axes: Axes, local_estimator: 'SpatialDFTBathyEstimator',
                                 sino1_fft: np.ndarray, sino2_fft: np.ndarray, kfft: np.ndarray,
                                 plt_min: float, plt_max: float, type: str, title: str,
                                 refinement_phase: bool = False) -> None:
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


def build_sinogram_fft_display(axes: Axes, title: str, values: np.ndarray, directions: np.ndarray,
                               kfft: np.ndarray, plt_min: float, plt_max: float, type: str,
                               ordonate: bool = True, abscissa: bool = True, **kwargs: dict) -> None:
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

