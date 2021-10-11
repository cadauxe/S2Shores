import os
from typing import TYPE_CHECKING

from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

import numpy as np


if TYPE_CHECKING:
    from ..local_bathymetry.temporal_correlation_bathy_estimator import \
        TemporalCorrelationBathyEstimator


def temporal_method_debug(temporal_estimator: 'TemporalCorrelationBathyEstimator') -> None:
    # FIXME : Handle severals wave_estimations
    ######################################################
    wave_estimation = temporal_estimator.waves_fields_estimations[0]
    ######################################################
    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(5, 3, figure=fig)

    # First diagram : first image of the sequence
    image = temporal_estimator.images_sequence[0].pixels
    imin = np.min(image)
    imax = np.max(image)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image, norm=Normalize(vmin=imin, vmax=imax))
    (l1, l2) = np.shape(image)
    radius = min(l1, l2) / 2
    ax.arrow(l1 // 2, l2 // 2, np.cos(np.deg2rad(wave_estimation.direction)) * (radius // 2),
             -np.sin(np.deg2rad(wave_estimation.direction)) * (radius // 2))
    plt.title('Thumbnail')

    # Second diagram : correlation matrix
    ax2 = fig.add_subplot(gs[0, 1])
    imin = np.min(temporal_estimator._correlation_matrix)
    imax = np.max(temporal_estimator._correlation_matrix)
    plt.imshow(temporal_estimator._correlation_matrix, norm=Normalize(vmin=imin, vmax=imax))
    (l1, l2) = np.shape(temporal_estimator._correlation_matrix)
    index = np.argmax(temporal_estimator.metrics['variances'])
    ax2.arrow(l1 // 2, l2 // 2,
              np.cos(np.deg2rad(wave_estimation.direction)) * (l1 // 4),
              -np.sin(np.deg2rad(wave_estimation.direction)) * (l1 // 4))
    plt.title('Correlation matrix')

    # Third diagram : Radon transform & maximum variance
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.imshow(temporal_estimator.radon_transform.get_sinograms_as_array(), interpolation='nearest',
               aspect='auto', origin='lower')
    (l1, l2) = np.shape(temporal_estimator.radon_transform.get_sinograms_as_array())
    plt.plot(l1 * temporal_estimator.metrics['variances'] /
             np.max(temporal_estimator.metrics['variances']),
             'r')
    ax3.arrow(index, 0, 0, l1)
    plt.annotate('%d °' % wave_estimation.direction, (index + 5, 10), color='orange')
    plt.title('Radon matrix')

    # Fourth diagram : Sinogram & wave length computation
    ax4 = fig.add_subplot(gs[2, :2])
    sinogram_max_var = temporal_estimator.metrics['sinogram_max_var']
    length_signal = len(sinogram_max_var)
    left_limit = max(int(length_signal / 2 - wave_estimation.wavelength / 2), 0)
    sinogram_period = temporal_estimator.metrics['sinogram_period']
    x = np.linspace(-length_signal // 2, length_signal // 2, length_signal)
    y = sinogram_max_var
    ax4.plot(x, y)
    ax4.plot(x[temporal_estimator.metrics['wave_length_zeros']],
             y[temporal_estimator.metrics['wave_length_zeros']], 'ro')
    ax4.annotate('L=%d m' % wave_estimation.wavelength,
                 (0, np.min(sinogram_max_var)),
                 color='r')
    ax4.arrow(
        x[int(length_signal / 2 + wave_estimation.wavelength /
              (2 * temporal_estimator.spatial_resolution))],
        np.min(sinogram_max_var), 0,
        np.abs(np.min(sinogram_max_var)) + np.max(
            sinogram_max_var), linestyle='dashed',
        color='g')
    ax4.arrow(
        x[int(length_signal / 2 - wave_estimation.wavelength /
              (2 * temporal_estimator.spatial_resolution))],
        np.min(sinogram_max_var), 0,
        np.abs(np.min(sinogram_max_var)) + np.max(
            sinogram_max_var), linestyle='dashed',
        color='g')
    argmax = np.argmax(temporal_estimator.metrics['sinogram_period'])
    ax4.plot(x[argmax + left_limit], sinogram_period[argmax], 'go')
    ax4.arrow(x[int(length_signal / 2)], 0,
              x[argmax + left_limit], 0, color='g')
    ax4.annotate('c = {:.2f} / {:.2f} = {:.2f} m/s'.format(temporal_estimator.metrics['dephasing'], temporal_estimator.metrics['delta_time'],
                                                           wave_estimation.celerity), (0, 0), color='orange')
    plt.title('Sinogram')

    # Fifth  diagram : Temporal reconstruction
    ax5 = fig.add_subplot(gs[3, :2])
    temporal_signal = temporal_estimator.metrics['temporal_signal']
    ax5.plot(temporal_signal)
    ax5.plot(temporal_estimator.metrics['arg_temporal_peaks_max'],
             temporal_signal[temporal_estimator.metrics['arg_temporal_peaks_max']], 'ro')
    ax5.annotate('T={:.2f} s'.format(wave_estimation.period),
                 (0, np.min(temporal_signal)), color='r')
    plt.title('Temporal reconstruction')
    fig.savefig(os.path.join(temporal_estimator.local_estimator_params.DEBUG_PATH,
                             f'Infos_point_{temporal_estimator.location[0]}_{temporal_estimator.location[1]}.png'), dpi=300)
