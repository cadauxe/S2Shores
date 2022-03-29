# -*- coding: utf-8 -*-
""" Class performing bathymetry computation using temporal correlation method

:author: Degoul Romain
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 18/06/2021
"""
import os

from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

import matplotlib as mpl
import numpy as np

from ..bathy_physics import depth_from_dispersion
from ..local_bathymetry.temporal_correlation_bathy_estimator import \
    TemporalCorrelationBathyEstimator

from .local_bathy_estimator_debug import LocalBathyEstimatorDebug


class TemporalCorrelationBathyEstimatorDebug(LocalBathyEstimatorDebug,
                                             TemporalCorrelationBathyEstimator):
    """ Class performing debugging for temporal correlation method
    """

    def explore_results(self) -> None:
        # FIXME: Handle severals wave_estimations
        ######################################################
        wave_estimation = self.waves_fields_estimations[0]
        wave_direction = wave_estimation.direction
        wave_wavelength = wave_estimation.wavelength
        wave_celerity = wave_estimation.celerity

        metrics = self.metrics
        # Note that wave direction is clockwise origin east
        px = np.cos(np.deg2rad(wave_direction))
        py = -np.sin(np.deg2rad(wave_direction))
        first_image = self.images_sequence[0].pixels
        correlation_matrix = self.correlation_image.pixels
        sinogram_max_var = metrics['sinogram_max_var']
        x = metrics['x_axis']
        interval = metrics['interval']
        debug_path = self.global_estimator.debug_path

        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(5, 2, figure=fig)

        # First diagram : first image of the sequence
        ax = fig.add_subplot(gs[0, 0])
        imin = np.min(first_image)
        imax = np.max(first_image)
        ax.imshow(first_image, norm=Normalize(vmin=imin, vmax=imax))
        (l1, l2) = np.shape(first_image)
        radius = min(l1, l2) / 3
        ax.arrow(l1 // 2, l2 // 2, radius * px, radius * py)
        plt.title('Thumbnail')

        # Second diagram : correlation matrix
        ax2 = fig.add_subplot(gs[0, 1])
        imin = np.min(correlation_matrix)
        imax = np.max(correlation_matrix)
        ax2.imshow(correlation_matrix, norm=Normalize(vmin=imin, vmax=imax))
        (l1, l2) = np.shape(correlation_matrix)
        radius = min(l1, l2) / 3
        ax2.arrow(l1 // 2, l2 // 2, radius * px, radius * py)
        plt.title('Correlation matrix')

        # Third diagram : Radon transform & maximum variance
        ax3 = fig.add_subplot(gs[1, :2])
        radon_array, _ = metrics['radon_transform'].get_as_arrays()
        s1, _ = radon_array.shape
        directions = self.selected_directions
        d1 = np.min(directions)
        d2 = np.max(directions)
        ax3.imshow(radon_array, interpolation='nearest', aspect='auto',
                   origin='lower', extent=[d1, d2, 0, s1])
        (l1, l2) = np.shape(radon_array)
        plt.plot(self.selected_directions, l1 * metrics['variances'] /
                 np.max(metrics['variances']), 'r')
        ax3.arrow(wave_direction, 0, 0, l1)
        plt.annotate('%d °' % wave_direction, (wave_direction + 5, 10), color='orange')
        plt.title('Radon matrix')

        # Fourth diagram : Sinogram & wave length computation
        ax4 = fig.add_subplot(gs[2, :2])

        ax4.plot(x, sinogram_max_var)
        ax4.scatter(x[interval], sinogram_max_var[interval], s=4 *
                    mpl.rcParams['lines.markersize'], c='orange')
        min_limit_x = np.min(x)
        min_limit_y = np.min(sinogram_max_var)
        ax4.plot(x[metrics['wave_length_zeros']],
                 sinogram_max_var[metrics['wave_length_zeros']], 'ro')
        ax4.plot(x[metrics['max_indices']],
                 sinogram_max_var[metrics['max_indices']], 'go')

        bathy = depth_from_dispersion(1 / wave_estimation.wavelength,
                                      wave_estimation.celerity, self.gravity)
        ax4.annotate('depth = {:.2f}'.format(bathy), (min_limit_x, min_limit_y), color='orange')
        plt.title('Sinogram')

        # Fifth  diagram
        ax5 = fig.add_subplot(gs[3, :2])
        ax5.axis('off')
        distances = metrics['distances']
        celerities = metrics['celerities']
        chain_dx = ' '.join([f'{distance:.2f} | ' for distance in distances])
        chain_celerities = ' '.join([f'{celerity:.2f} | ' for celerity in celerities])
        chain_coefficients = ' '.join(
            [f'{coefficient:.2f} | ' for coefficient in metrics['linearity_coefficients']])
        ax5.annotate(f'wave_length = {wave_wavelength} \n dx = {chain_dx} \n c = {chain_celerities} \n'
                     f' ckg = {chain_coefficients}\n'
                     f' chosen_celerity = {wave_celerity}', (0, 0), color='g')
        fig.savefig(os.path.join(
            debug_path, f'Infos_point_{self.location[0]}_{self.location[1]}.png'), dpi=300)
