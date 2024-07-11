# -*- coding: utf-8 -*-
"""
Class managing the computation of wave fields from two images taken at a small time interval.


:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2024 CNES. All rights reserved.
:created: 5 March 2021
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
from typing import TYPE_CHECKING, List, Optional, Tuple  # @NoMove

import cmcrameri.cm as cmc
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..image_processing.waves_radon import WavesRadon
from .bathy_visualization.display_utils import (get_display_title,
                                                get_display_title_with_kernel)
from .bathy_visualization.sinogram_display import (build_sinogram_fft_display,
                                                   animate_sinograms)
from .bathy_visualization.wave_images_display import build_display_waves_image

if TYPE_CHECKING:
    from ..local_bathymetry.spatial_correlation_bathy_estimator import (
        SpatialCorrelationBathyEstimator)  # @UnusedImport
    from ..local_bathymetry.spatial_dft_bathy_estimator import (
        SpatialDFTBathyEstimator)  # @UnusedImport


def build_image_display(axes: Axes, title: str, image: np.ndarray,
                        directions: Optional[List[Tuple[float, float]]] = None,
                        cmap: Optional[str] = None) -> None:
    """ Build an image display"""
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
    """ Build a 2D display with given directions"""
    extent = [np.min(directions), np.max(directions), 0, values.shape[0]]
    imin = np.min(values)
    imax = np.max(values)
    axes.imshow(values, norm=Normalize(vmin=imin, vmax=imax), extent=extent, **kwargs)
    axes.set_xticks(directions[::40])
    plt.setp(axes.get_xticklabels(), fontsize=8)
    axes.set_title(title, fontsize=10)


def build_directional_curve_display(axes: Axes, title: str,
                                    values: np.ndarray, directions: np.ndarray) -> None:
    """ Build a curve display with given directions"""
    axes.plot(directions, values)
    axes.set_xticks(directions[::20])
    plt.setp(axes.get_xticklabels(), fontsize=8)
    axes.set_title(title)


def display_initial_data(local_estimator: 'SpatialDFTBathyEstimator') -> None:
    """ Display the initial data of the object local_estimator"""
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


def build_correl_spectrum_matrix_spatial_correlation(
        axes: Axes,
        local_estimator: 'SpatialCorrelationBathyEstimator',
        sino1_fft: np.ndarray,
        sino2_fft: np.ndarray,
        kfft: np.ndarray,
        type: str,
        title: str,
        refinement_phase: bool=False) -> None:
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


def build_polar_display(fig: Figure, axes: Axes, title: str,
                        local_estimator: 'SpatialDFTBathyEstimator',
                        values: np.ndarray, resolution: float, dfn_max: float, max_wvlgth: float,
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


def build_radon_transform_display(axs: Axes, transform: WavesRadon, title: str,
                                  refinement_phase: bool=False) -> None:
    values, directions = transform.get_as_arrays()
    sino_fft = transform.get_sinograms_standard_dfts()
    dft_amplitudes = np.abs(sino_fft)
    dft_phases = np.angle(sino_fft)
    variances = transform.get_sinograms_variances()

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
    display_energies(local_estimator, radon1, radon2)
    animate_sinograms(local_estimator, radon1, radon2)
