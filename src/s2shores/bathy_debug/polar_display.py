# -*- coding: utf-8 -*-
"""
Class managing the polar display.

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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure


if TYPE_CHECKING:
    from ..local_bathymetry.spatial_dft_bathy_estimator import (
        SpatialDFTBathyEstimator)  # @UnusedImport

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

