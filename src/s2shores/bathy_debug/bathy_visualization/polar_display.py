# -*- coding: utf-8 -*-
"""
Functions for polar plot displays

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
import os
from typing import List, Optional, Tuple  # @NoMove

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure

import s2shores.bathy_debug.bathy_visualization.display_utils as display_utils
import s2shores.bathy_debug.bathy_visualization.wave_images_display as wave_images_display


def build_polar_plot(fig: Figure, axes: Axes, title: str, image: np.ndarray,
                              resolution: float,
                              subplot_pos: [float, float, float],
                              directions: Optional[List[Tuple[float, float]]] = None,
                              cmap: Optional[str] = None, coordinates: bool=True,
                     polar_labels = ('0°', '90°', '180°', '-90°')) -> None:
    (l1, l2) = np.shape(image)
    imin = np.min(image)
    imax = np.max(image)
    axes.imshow(image, norm=Normalize(vmin=imin, vmax=imax), cmap=cmap)
    # create polar axes in the foreground and remove its background to see through
    subplot_locator = int(f'{subplot_pos[0]}{subplot_pos[1]}{subplot_pos[2]}')
    ax_polar = fig.add_subplot(subplot_locator, polar=True)
    ax_polar.set_yticklabels([])
    polar_ticks = np.arange(len(polar_labels)) * np.pi / 2.

    plt.xticks(polar_ticks, polar_labels, size=9, color='blue')
    for i, label in enumerate(ax_polar.get_xticklabels()):
        label.set_rotation(i * 45)
    ax_polar.set_facecolor('None')

    xmax = f'{l1}px \n {np.round((l1-1)*resolution)}m'
    axes.set_xticks([0, l1 - 1], ['0', xmax], fontsize=8)
    ymax = f'{l2}px \n {np.round((l2-1)*resolution)}m'

    if coordinates:
        axes.set_yticks([0, l2 - 1], ['0', ymax], fontsize=8)
    else:
        axes.set_yticks([0, l2 - 1], ['', ''], fontsize=8)
        axes.set_xticks([0, l1 - 1], ['\n', ' \n'], fontsize=8)

    if directions is not None:
        # Normalization of arrows length
        coeff_length_max = np.max((list(zip(*directions))[1]))
        radius = np.floor(min(l1, l2) / 2) - 5
        for direction, coeff_length in directions:
            arrow_length = radius * coeff_length / coeff_length_max
            dir_rad = np.deg2rad(direction)
            axes.arrow(l1 // 2, l2 // 2,
                       np.cos(dir_rad) * arrow_length,
                       -np.sin(dir_rad) * arrow_length,
                       head_width=2, head_length=3, color='r')

    axes.xaxis.tick_top()
    axes.set_title(title, fontsize=9, loc='center')

def build_polar_display(fig: Figure, axes: Axes, title: str,
                        local_estimator: 'SpatialDFTBathyEstimator',
                        values: np.ndarray, resolution: float, dfn_max: float, max_wvlgth: float,
                        subplot_pos: [float, float, float],
                        refinement_phase: bool=False, **kwargs: dict) -> None:
    """Build a polar display with the given parameters."""
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
    direc_from_north = dfn_max
    main_direction = 270 - dfn_max
    main_wavelength = max_wvlgth

    estimations = local_estimator.bathymetry_estimations
    sorted_estimations_args = estimations.argsort_on_attribute(
        local_estimator.final_estimations_sorting)
    delta_time = estimations.get_estimations_attribute('delta_time')[sorted_estimations_args[0]]
    delta_phase = estimations.get_estimations_attribute('delta_phase')[sorted_estimations_args[0]]

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

    ax_polar.plot(np.radians((main_direction + 180) % 360), 1 / main_wavelength, '*', color='black')

    ax_polar.annotate('Peak at \n[$\Theta$={:.1f}°, \n$\lambda$={:.2f}m]'.format((direc_from_north), main_wavelength),
                      xy=[np.radians(main_direction % 180), (1 / main_wavelength)],  # theta, radius
                      xytext=(0.5, 0.65),    # fraction, fraction
                      textcoords='figure fraction',
                      horizontalalignment='left',
                      verticalalignment='bottom',
                      fontsize=10, color='blue')

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

    # convert the direction coordinates in the polar plot axis (from
    directions = (directions + 180) % 360
    # Add the last element of the list to the list.
    # This is necessary or the line from 330 deg to 0 degree does not join up on the plot.
    ddir = np.diff(directions).mean()
    directions = np.append(directions, directions[-1:] + ddir)

    plotval = np.concatenate((plotval, plotval[:, 0].reshape(plotval.shape[0], 1)), axis=1)

    a, r = np.meshgrid(np.deg2rad(directions), wavenumbers)
    tcf = ax_polar.tricontourf(a.flatten(), r.flatten(), plotval.flatten(), 500, cmap='gist_ncar_r')
    plt.colorbar(tcf, ax=ax_polar)

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
    fig.suptitle(display_utils.get_display_title_with_kernel(local_estimator), fontsize=12)

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
    wave_images_display.build_display_waves_image(
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
