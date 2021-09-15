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
    ax.arrow(l1 // 2, l2 // 2, np.cos(np.deg2rad(temporal_estimator.direction_propagation)) * (radius // 2),
             -np.sin(np.deg2rad(temporal_estimator.direction_propagation)) * (radius // 2))
    plt.title('Thumbnail')

    # Second diagram : correlation matrix
    ax2 = fig.add_subplot(gs[0, 1])
    plt.imshow(temporal_estimator._correlation_matrix)
    (l1, l2) = np.shape(temporal_estimator._correlation_matrix)
    index = np.argmax(temporal_estimator._variance)
    ax2.arrow(l1 // 2, l2 // 2,
              np.cos(np.deg2rad(temporal_estimator.direction_propagation)) * (l1 // 4),
              -np.sin(np.deg2rad(temporal_estimator.direction_propagation)) * (l1 // 4))
    plt.title('Correlation matrix')

    # Third diagram : Radon transform & maximum variance
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.imshow(temporal_estimator.radon_transform.get_sinograms_as_array(), interpolation='nearest',
               aspect='auto', origin='lower')
    (l1, l2) = np.shape(temporal_estimator.radon_transform.get_sinograms_as_array())
    plt.plot(l1 * temporal_estimator._variance / np.max(temporal_estimator._variance), 'r')
    ax3.arrow(index, 0, 0, l1)
    plt.annotate('%d °' % temporal_estimator.direction_propagation, (index + 5, 10), color='orange')
    plt.title('Radon matrix')

    # Fourth diagram : Sinogram & wave length computation
    ax4 = fig.add_subplot(gs[2, :2])
    length_signal = len(temporal_estimator.sinogram_max_var.sinogram.flatten())
    left_limit = max(int(length_signal / 2 - temporal_estimator.wave_length / 2), 0)
    right_limit = min(int(length_signal / 2 + temporal_estimator.wave_length / 2), length_signal)
    signal_period = temporal_estimator.sinogram_max_var.sinogram.flatten()[left_limit:right_limit]
    x = np.linspace(-length_signal // 2, length_signal // 2, length_signal)
    y = temporal_estimator._sinogram_max_var.sinogram.flatten()
    ax4.plot(x, y)
    ax4.plot(x[temporal_estimator._wave_length_zeros],
             y[temporal_estimator._wave_length_zeros], 'ro')
    ax4.annotate('L=%d m' % temporal_estimator.wave_length,
                 (0, np.min(temporal_estimator._sinogram_max_var.sinogram.flatten())),
                 color='r')
    ax4.arrow(
        x[int(length_signal / 2 + temporal_estimator.wave_length /
              (2 * temporal_estimator.local_estimator_params.RESOLUTION.SPATIAL))],
        np.min(temporal_estimator._sinogram_max_var.sinogram.flatten()), 0,
        np.abs(np.min(temporal_estimator._sinogram_max_var.sinogram.flatten())) + np.max(
            temporal_estimator._sinogram_max_var.sinogram.flatten()), linestyle='dashed',
        color='g')
    ax4.arrow(
        x[int(length_signal / 2 - temporal_estimator.wave_length /
              (2 * temporal_estimator.local_estimator_params.RESOLUTION.SPATIAL))],
        np.min(temporal_estimator._sinogram_max_var.sinogram.flatten()), 0,
        np.abs(np.min(temporal_estimator._sinogram_max_var.sinogram.flatten())) + np.max(
            temporal_estimator._sinogram_max_var.sinogram.flatten()), linestyle='dashed',
        color='g')
    argmax = np.argmax(temporal_estimator._signal_period)
    ax4.plot(x[argmax + left_limit], signal_period[argmax], 'go')
    ax4.arrow(x[int(length_signal / 2)], 0,
              x[argmax + left_limit], 0, color='g')
    ax4.annotate('c = {:.2f} / {:.2f} = {:.2f} m/s'.format(temporal_estimator._dephasing, temporal_estimator._duration,
                                                           temporal_estimator.celerity), (0, 0), color='orange')
    plt.title('Sinogram')

    # Fifth  diagram : Temporal reconstruction
    ax5 = fig.add_subplot(gs[3, :2])
    ax5.plot(temporal_estimator._temporal_signal)
    ax5.plot(temporal_estimator._temporal_arg_peaks_max,
             temporal_estimator._temporal_signal[temporal_estimator._temporal_arg_peaks_max], 'ro')
    ax5.annotate('T={:.2f} s'.format(temporal_estimator.period),
                 (0, np.min(temporal_estimator._temporal_signal)), color='r')
    plt.title('Temporal reconstruction')
    fig.savefig(os.path.join(temporal_estimator.local_estimator_params.DEBUG_PATH,
                             f'Infos_point_{temporal_estimator._position[0]}_{temporal_estimator._position[1]}.png'), dpi=300)
