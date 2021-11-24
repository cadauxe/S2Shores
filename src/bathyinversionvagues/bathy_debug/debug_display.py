# -*- coding: utf-8 -*-
import os
from typing import TYPE_CHECKING  # @NoMove

from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl
from ..bathy_physics import funLinearC_k

import numpy as np


if TYPE_CHECKING:
    from ..local_bathymetry.temporal_correlation_bathy_estimator import \
        TemporalCorrelationBathyEstimator


def temporal_method_debug(temporal_estimator: 'TemporalCorrelationBathyEstimator') -> None:
    # FIXME : Handle severals wave_estimations
    ######################################################
    wave_estimation = temporal_estimator.waves_fields_estimations[0]
    wave_direction = wave_estimation.direction
    wave_wavelength = wave_estimation.wavelength
    wave_celerity = wave_estimation.celerity
    wave_period = wave_estimation.period
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
    ax.arrow(l1 // 2, l2 // 2, np.cos(np.deg2rad(wave_direction)) * (radius // 2),
             -np.sin(np.deg2rad(wave_direction)) * (radius // 2))
    plt.title('Thumbnail')

    # Second diagram : correlation matrix
    ax2 = fig.add_subplot(gs[0, 1])
    correlation_matrix = temporal_estimator.correlation_image.pixels
    imin = np.min(correlation_matrix)
    imax = np.max(correlation_matrix)
    plt.imshow(correlation_matrix, norm=Normalize(vmin=imin, vmax=imax))
    (l1, l2) = np.shape(correlation_matrix)
    index = np.argmax(temporal_estimator.metrics['variances'])
    ax2.arrow(l1 // 2, l2 // 2,
              np.cos(np.deg2rad(wave_direction)) * (l1 // 4),
              -np.sin(np.deg2rad(wave_direction)) * (l1 // 4))
    plt.title('Correlation matrix')

    # Third diagram : Radon transform & maximum variance
    ax3 = fig.add_subplot(gs[1, :2])
    radon_array, _ = temporal_estimator.radon_transform.get_as_arrays()
    ax3.imshow(radon_array, interpolation='nearest', aspect='auto', origin='lower')
    (l1, l2) = np.shape(radon_array)
    plt.plot(
        l1 *
        temporal_estimator.metrics['variances'] /
        np.max(
            temporal_estimator.metrics['variances']),
        'r')
    ax3.arrow(index, 0, 0, l1)
    plt.annotate('%d °' % wave_direction, (index + 5, 10), color='orange')
    plt.title('Radon matrix')

    # Fourth diagram : Sinogram & wave length computation
    ax4 = fig.add_subplot(gs[2, :2])
    sinogram_max_var = temporal_estimator.metrics['sinogram_max_var']
    length_signal = len(sinogram_max_var)
    left_limit = max(int(length_signal / 2 - wave_wavelength / 2), 0)
    x = np.linspace(-length_signal // 2, length_signal // 2, length_signal)
    y = sinogram_max_var
    ax4.plot(x, y)
    ax4.scatter(x[temporal_estimator._metrics['interval']],
                y[temporal_estimator._metrics['interval']], s=4 * mpl.rcParams['lines.markersize'],
                c='orange')
    min_limit_x = np.min(x)
    min_limit_y = np.min(y)
    ax4.plot(x[temporal_estimator._metrics['wave_length_zeros']],
             y[temporal_estimator._metrics['wave_length_zeros']], 'ro')
    ax4.plot(x[temporal_estimator._metrics['max_indice']],
             y[temporal_estimator._metrics['max_indice']], 'go')

    bathy = funLinearC_k(1 / wave_estimation.wavelength, wave_estimation.celerity, 0.01, 9.8)
    ax4.annotate('depth = {:.2f}'.format(bathy), (min_limit_x, min_limit_y), color='orange')
    plt.title('Sinogram')

    # Fifth  diagram : Temporal reconstruction
    ax5 = fig.add_subplot(gs[3, :2])
    temporal_signal = temporal_estimator.metrics['temporal_signal']
    ax5.plot(temporal_signal)
    ax5.plot(temporal_estimator.metrics['arg_temporal_peaks_max'],
             temporal_signal[temporal_estimator.metrics['arg_temporal_peaks_max']], 'ro')
    ax5.annotate('T={:.2f} s'.format(wave_period),
                 (0, np.min(temporal_signal)), color='r')
    plt.title('Temporal reconstruction')
    ax5.axis('off')
    ax5.annotate('wave_length = %d \n dx = |dx| = %d \n nb_l = %d \n propagated distance =dx + nb_l*wave_length = %d m \n t_offshore = %f \n c = %f / %f = %f m/s' % (wave_estimation.wavelength, temporal_estimator._metrics['dx'], temporal_estimator._metrics['nb_l'], temporal_estimator._metrics['dephasing'], temporal_estimator._metrics['t_offshore'], temporal_estimator._metrics['dephasing'], temporal_estimator._metrics['propagation_duration'], wave_estimation.celerity),
                 (0, 0), color='g')

    print('PATH')
    print(os.path.join(temporal_estimator.local_estimator_params.DEBUG_PATH))
    fig.savefig(os.path.join(temporal_estimator.local_estimator_params.DEBUG_PATH,
                             f'Infos_point_{temporal_estimator.location[0]}_{temporal_estimator.location[1]}.png'), dpi=300)
