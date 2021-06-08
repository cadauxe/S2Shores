# -*- coding: utf-8 -*-
"""
Class managing the computation of waves fields from two images taken at a small time interval.


:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 5 mars 2021
"""
import os

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def display_curve(data, legend):
    _, ax = plt.subplots()
    ax.plot(data)
    ax.set_title(legend)
    plt.show()


def display_3curves(data1, data2, data3):
    _, ax = plt.subplots(3)
    ax[0].plot(data1)
    ax[1].plot(data2)
    ax[2].plot(data3)
    plt.show()


def display_4curves(data1, data2, data3, data4):
    _, ax = plt.subplots(2, 2)
    ax[0, 0].plot(data1)
    ax[1, 0].plot(data2)
    ax[0, 1].plot(data3)
    ax[1, 1].plot(data4)
    plt.show()


def display_image(data, legend):
    _, ax = plt.subplots()
    ax.imshow(data, aspect='auto', cmap='gray')
    ax.set_title(legend)
    plt.show()


def display_estimation(amplitude, amplitude_sino1, phase,
                       phase_thresholded, totspec, totalSpecMax_ref):
    plt.close('all')
    _, axs = plt.subplots(2, 3)
    axs[0, 1].imshow(amplitude, aspect='auto', cmap='gray')
    axs[0, 1].set_title('Combined Amplitude')
    axs[1, 1].imshow(amplitude_sino1, aspect='auto', cmap='gray')
    axs[1, 1].set_title('Amplitude Sino1')
    axs[1, 0].imshow(phase, aspect='auto', cmap='gray')
    axs[1, 0].set_title('phase shift')
    axs[0, 0].imshow(phase_thresholded, aspect='auto', cmap='gray')
    axs[0, 0].set_title('phase shift thresholded')
    axs[0, 2].imshow(totspec, aspect='auto', cmap='gray')
    axs[0, 2].set_title('totspec')
    axs[1, 2].plot(totalSpecMax_ref)
    # axs[0, 2].plot(sinograms2_energies / image2_energy)
    axs[1, 2].set_title('totalSpecMax_ref')
    plt.show()


def draw_results(Im, angle, corr_car, radon_matrix, variance, sinogram_max_var, sinogram_tuned, argmax,
                 wave_length_peaks, wave_length, config, celerity, peaks_max, SS_filtered, T, path):
    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(5, 4, figure=fig)
    imin = np.min(Im[:, :, 0])
    imax = np.max(Im[:, :, 0])
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(Im[:, :, 0], norm=matplotlib.colors.Normalize(vmin=imin, vmax=imax))
    (l1, l2, l3) = np.shape(Im)
    radius = min(l1, l2) / 2
    ax.arrow(l1 // 2, l2 // 2, np.cos(np.deg2rad(angle)) * (radius // 2),
             -np.sin(np.deg2rad(angle)) * (radius // 2))
    plt.title('Thumbnail')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(Im[:, :, 0], norm=matplotlib.colors.Normalize(vmin=imin, vmax=imax))
    (l1, l2, l3) = np.shape(Im)
    ax1.arrow(l1 // 2, l2 // 2, np.cos(np.deg2rad(angle)) * (radius // 2),
              -np.sin(np.deg2rad(angle)) * (radius // 2))
    plt.title('Thumbnail filtered')

    ax2 = fig.add_subplot(gs[0, 2])
    plt.imshow(corr_car)
    (l1, l2) = np.shape(corr_car)
    ax2.arrow(l1 // 2, l2 // 2,
              np.cos(np.deg2rad(angle)) * (l1 // 4), -np.sin(np.deg2rad(angle)) * (l1 // 4))
    plt.title('Correlation matrix')

    ax3 = fig.add_subplot(gs[1, :3])
    ax3.imshow(radon_matrix, interpolation='nearest', aspect='auto', origin='lower')
    (l1, l2) = np.shape(radon_matrix)
    plt.plot(l1 * variance / np.max(variance), 'r')
    ax3.arrow(angle, 0, 0, l1)
    plt.annotate('%d °' % angle, (angle + 5, 10), color='orange')
    plt.title('Radon matrix')

    ax4 = fig.add_subplot(gs[2, :3])
    length_signal = len(sinogram_tuned)
    x = np.linspace(-length_signal // 2, length_signal // 2, length_signal)
    ax4.plot(x, sinogram_max_var, '--')
    ax4.plot(x, sinogram_tuned)
    ax4.plot(x[wave_length_peaks], sinogram_tuned[wave_length_peaks], 'ro')
    ax4.annotate('L=%d m' % wave_length, (0, np.min(sinogram_tuned)), color='r')
    ax4.arrow(x[int(length_signal / 2 + wave_length / (2 * config.TEMPORAL_METHOD.RESOLUTION.SPATIAL))],
              np.min(sinogram_tuned), 0,
              np.abs(np.min(sinogram_tuned)) + np.max(sinogram_tuned), linestyle='dashed', color='g')
    ax4.arrow(x[int(length_signal / 2 - wave_length / (2 * config.TEMPORAL_METHOD.RESOLUTION.SPATIAL))],
              np.min(sinogram_tuned), 0,
              np.abs(np.min(sinogram_tuned)) + np.max(sinogram_tuned), linestyle='dashed', color='g')
    ax4.plot(x[int(argmax)], sinogram_tuned[int(argmax)], 'go')
    ax4.arrow(x[int(length_signal / 2)], 0,
              argmax - len(sinogram_tuned) / (2 * config.TEMPORAL_METHOD.RESOLUTION.SPATIAL), 0, color='g')
    ax4.annotate('c = {:.2f} / {:.2f} = {:.2f} m/s'.format(
        (argmax - len(sinogram_tuned) / (2 * config.TEMPORAL_METHOD.RESOLUTION.SPATIAL)),
        config.TEMPORAL_METHOD.TEMPORAL_LAG * config.TEMPORAL_METHOD.RESOLUTION.TEMPORAL, celerity), (
        x[int(argmax - wave_length / (2 * config.TEMPORAL_METHOD.RESOLUTION.SPATIAL) + length_signal / 2)],
        np.max(sinogram_tuned) - 10), color='orange')
    plt.title('Sinogram')

    ax5 = fig.add_subplot(gs[3, :3])
    ax5.plot(SS_filtered)
    ax5.plot(peaks_max, SS_filtered[peaks_max], 'ro')
    ax5.annotate('T={:.2f} s'.format(T), (0, np.min(SS_filtered)), color='r')
    plt.title('Temporal reconstruction')
    fig.savefig(os.path.join(path, 'Infos_point.png'), dpi=300)
