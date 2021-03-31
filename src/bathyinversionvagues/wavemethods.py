#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 3 10:12:00 2021

Module containing all wave parameters estimation methods

@author: erwinbergsma
         gregoirethoumyre
         degoulromain
"""

# Imports
import copy
import os
from scipy.signal import find_peaks

import matplotlib

from bathyinversionvagues.shoresutils import (sc_all, funDetrend_2d, funGetSpectralPeaks, DFT_fr,
                                              radon, fft_filtering, compute_sinogram,
                                              create_sequence_time_series_temporal,
                                              compute_temporal_correlation, compute_celerity,
                                              cartesian_projection, correlation_tuning,
                                              sinogram_tuning, compute_wave_length, compute_period,
                                              temporal_reconstruction,
                                              temporal_reconstruction_tuning,
                                              create_sequence_time_series_spatial,
                                              compute_angles_distances, compute_spatial_correlation)
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def spatial_dft_method(Im, params, kfft, phi_min, phi_deep):
    """
    issu de shores.py - funRadonSpecCel_v3_3
    VERSION 3.3 --  QUICKER:    Radon, FFT, find directional peaks, then do
                                detailed DFT analysis to find detailed phase
                                shifts per linear wave number (k *2pi)


    Parameters
    ----------
    Im : numpy.ndarray
        Sub-windowed images in M x N x BANDS -- currently only 2 bands are used!
    kfft : numpy.ndarray
        M x 1 array with LINEAR wave number.
    phi_min : numpy.ndarray
        M x N of size(kfft,theta) that contains the minimum phase shift based on shallow water.
    phi_deep : numpy.ndarray
        M x N of size(kfft,theta) that contains the maximum phase shift based on deep water.
    params : dictionary
        This dictionary contains settings and image specifics.


     Returns
    -------
    dict:
        As output we deliver a dictionary containing
            -   cel     =   Wave celerity               [m/s]
            -   nu       =   linear Wave number                 [1/m]
            -   L       =   Wavelength                  [m]
            -   T       =   Approximate wave period     [sec]
            -   dir     =   Wave direction (RADON)      [degrees]
            -   dPhi    =   Measured phase shift        [rad]

    """

    # pre-set all output variables:)
    kKeep = params.NKEEP
    CEL = np.empty(kKeep) * np.nan  # Estimated celerity
    DIR = np.empty(kKeep) * np.nan
    dPHI = np.empty(kKeep) * np.nan
    PHIRat = np.empty(kKeep) * np.nan
    NU = np.empty(kKeep) * np.nan
    Emax = np.empty(kKeep) * np.nan
    T = np.empty(kKeep) * np.nan
    DCEL = np.empty(kKeep) * np.nan

    # All rotational angles (theta) for the Radon Transform
    thetaFFT = np.linspace(params.MIN_DIR, params.MAX_DIR, params.MAX_NDIRS, endpoint=False)

    # Check if the image is NOT empty (if statement):
    if sc_all(Im):
        # Create Radon sinograms per sub image [we can make this a for loop for
        # the number of frames]
        sinogram1 = radon(funDetrend_2d(Im[:, :, 0]), theta=thetaFFT)
        sinogram2 = radon(funDetrend_2d(Im[:, :, 1]), theta=thetaFFT)
        # signal length to normalise the spectrum:
        N = sinogram1.shape[0]
        # Retrieve total spectrum, controlled by physical wave propagatinal limits:
        totalSpecFFT, _, _, phase_check = funGetSpectralPeaks(Im, thetaFFT, params.UNWRAP_PHASE_SHIFT, params.DT,
                                                              params.DX, params.MIN_D, params.G)
        # Find maximum total energy per direction theta
        totalSpecMaxheta = np.max(totalSpecFFT, axis=0) / np.max(np.max(totalSpecFFT, axis=0))
        # Pick the maxima
        peaksDir = find_peaks(totalSpecMaxheta, prominence=params.PROMINENCE_MAX_PEAK)

        if peaksDir[0].size > 0:
            for ii in range(0, peaksDir[0].size):
                tmp = np.arange(np.max([peaksDir[0][ii] - params.ANGLE_AROUND_PEAK_DIR, 0]),
                                np.min([peaksDir[0][ii] + params.ANGLE_AROUND_PEAK_DIR + 1, 360]))
                if ii == 0:
                    dirInd = tmp
                else:
                    dirInd = np.append(dirInd, tmp)

            # delete double directions:
            dirInd = np.unique(dirInd)
            # create phi limits matrix
            phi_min = np.tile(phi_min[:, np.newaxis], (1, dirInd.shape[0]))
            phi_deep = np.tile(phi_deep[:, np.newaxis], (1, dirInd.shape[0]))
            thetaTmp = thetaFFT[dirInd]
            # Detailed analysis of the signal for positive phase shifts
            sinoFFT1 = np.empty((kfft.size, dirInd.shape[0])) * (np.nan + 0.j)
            sinoFFT2 = np.empty((kfft.size, dirInd.shape[0])) * (np.nan + 0.j)

            for ii in range(0, dirInd.shape[0]):
                sinoFFT1[:, ii] = DFT_fr(sinogram1[:, dirInd[ii]], kfft, 1 / params.DX)
                sinoFFT2[:, ii] = DFT_fr(sinogram2[:, dirInd[ii]], kfft, 1 / params.DX)

            sinoFFt = np.dstack((sinoFFT1, sinoFFT2))
            # This allows to calucalate the phase, amplitude:
            phase_shift = np.angle(sinoFFt[:, :, 1] * np.conj(sinoFFt[:, :, 0]))
            # the phase comes between -pi and pi but we want to know the fraction of
            # the total wave thus  0 < dphi < 2pi
            phase_shift_unw = copy.deepcopy(phase_shift)

            if not params.UNWRAP_PHASE_SHIFT:
                # currently deactivated but we want this functionality:
                phase_shift_unw = np.abs(phase_shift_unw)
            else:
                phase_shift_unw = (phase_shift_unw + 2 * np.pi) % (2 * np.pi)

            # Deep water limitation [if the wave travels faster that the deep-water
            # limit we consider it non-physical]
            phase_shift[phase_shift_unw > phi_deep[:, :dirInd.shape[0]]] = 0
            phase_shift_unw[phase_shift_unw > phi_deep[:, :dirInd.shape[0]]] = 0
            # Minimal propagation speed; this depends on the Satellite; Venus or Sentinel 2
            phase_shift[phase_shift_unw < phi_min[:, :dirInd.shape[0]]] = 0
            phase_shift_unw[phase_shift_unw < phi_min[:, :dirInd.shape[0]]] = 0
            totSpec = (np.abs(
                ((np.abs(sinoFFt[:, :, 0]) ** 2 + np.abs(sinoFFt[:, :, 1]) ** 2) / (N ** 2)) * phase_shift_unw) / N)
            # Refined spectral solution:
            totalSpecMax_ref = np.max(totSpec, axis=0) / np.max(np.max(totSpec, axis=0))
            peaksDir = find_peaks(totalSpecMax_ref, prominence=params.PROMINENCE_MULTIPLE_PEAKS)
            peaksDir = peaksDir[0][:kKeep]

            if peaksDir.size > 0:
                Emax[:len(peaksDir)] = totalSpecMax_ref[peaksDir]
                DIR[:len(peaksDir)] = thetaTmp[peaksDir]
                peaksK = np.argmax(totSpec[:, peaksDir], axis=0)
                NU[:len(peaksDir)] = kfft[peaksK].squeeze()
                dPHI[:len(peaksDir)] = phase_shift_unw[peaksK, peaksDir]
                PHIRat[:len(peaksDir)] = dPHI[:len(peaksDir)] / phi_deep[peaksK, peaksDir]
                CEL = dPHI / (2 * np.pi * NU * params.DT)
                T = 1 / (CEL * NU)

                for ii in range(0, np.min((DIR.shape[0], kKeep))):
                    if (dPHI[ii] != 0) or (not np.isnan(dPHI[ii])):
                        if (T[ii] <= params.MIN_T) or (T[ii] >= params.MAX_T):
                            NU[ii] = np.nan
                            DIR[ii] = np.nan
                            CEL[ii] = np.nan
                            DCEL[ii] = np.nan
                            T[ii] = np.nan
                            PHIRat[ii] = np.nan

                # sort now on longest waves:
                sorting = np.argsort(-((PHIRat ** 2) * Emax))

                CEL = CEL[sorting]
                DCEL = DCEL[sorting]
                NU = NU[sorting]
                T = T[sorting]
                DIR = DIR[sorting]

    return {'cel': CEL,
            'nu': NU,
            'T': T,
            'dir': DIR,
            'dcel': DCEL
            }


def temporal_correlation_method(Im, config):
    """
    Bathymetry computation function based on time series correlation

    Parameters
    ----------
    Im : numpy.ndarray
        Sub-windowed images in M x N x BANDS
     Returns
    -------
    dict:
        As output we deliver a dictionary containing
            -   cel     =   Wave celerity               [m/s]
            -   nu       =   linear Wave number                 [1/m]
            -   L       =   Wavelength                  [m]
            -   T       =   Approximate wave period     [sec]
            -   dir     =   Wave direction (RADON)      [degrees]

    """
    try:
        if config.TEMPORAL_METHOD.PASS_BAND_FILTER:
            Im, flag = fft_filtering(Im, config.TEMPORAL_METHOD.RESOLUTION.SPATIAL,
                                     T_max=config.PREPROCESSING.PASSBAND.HIGH_PERIOD,
                                     T_min=config.PREPROCESSING.PASSBAND.LOW_PERIOD)
        stime_series, xx, yy = create_sequence_time_series_temporal(Im=Im,
                                                                    percentage_points=config.TEMPORAL_METHOD.PERCENTAGE_POINTS)
        corr = compute_temporal_correlation(sequence_thumbnail=stime_series,
                                            number_frame_shift=config.TEMPORAL_METHOD.TEMPORAL_LAG)
        corr_car, distances, angles = cartesian_projection(corr_matrix=corr, xx=xx, yy=yy,
                                                           spatial_resolution=config.TEMPORAL_METHOD.RESOLUTION.SPATIAL)
        corr_car_tuned = correlation_tuning(correlation_matrix=corr_car,
                                            ratio=config.TEMPORAL_METHOD.TUNING.RATIO_SIZE_CORRELATION)
        (sinogram_max_var, angle, variance, radon_matrix) = compute_sinogram(correlation_matrix=corr_car_tuned,
                                                                             median_filter_kernel_ratio=config.TEMPORAL_METHOD.TUNING.MEDIAN_FILTER_KERNEL_RATIO_SINOGRAM,
                                                                             mean_filter_kernel_size=config.TEMPORAL_METHOD.TUNING.MEAN_FILTER_KERNEL_SIZE_SINOGRAM)
        sinogram_tuned = sinogram_tuning(sinogram=sinogram_max_var,
                                         mean_filter_kernel_size=config.TEMPORAL_METHOD.TUNING.MEAN_FILTER_KERNEL_SIZE_SINOGRAM)
        wave_length, wave_length_peaks = compute_wave_length(sinogram=sinogram_tuned)
        celerity, argmax = compute_celerity(sinogram=sinogram_tuned, wave_length=wave_length,
                                            spatial_resolution=config.TEMPORAL_METHOD.RESOLUTION.SPATIAL,
                                            time_resolution=config.TEMPORAL_METHOD.RESOLUTION.TEMPORAL,
                                            temporal_lag=config.TEMPORAL_METHOD.TEMPORAL_LAG)
        SS = temporal_reconstruction(angle=angle, angles=np.degrees(angles), distances=distances, celerity=celerity,
                                     correlation_matrix=corr,
                                     time_interpolation_resolution=config.TEMPORAL_METHOD.RESOLUTION.TIME_INTERPOLATION)
        SS_filtered = temporal_reconstruction_tuning(SS,
                                                     time_interpolation_resolution=config.TEMPORAL_METHOD.RESOLUTION.TIME_INTERPOLATION,
                                                     low_frequency_ratio=config.TEMPORAL_METHOD.TUNING.LOW_FREQUENCY_RATIO_TEMPORAL_RECONSTRUCTION,
                                                     high_frequency_ratio=config.TEMPORAL_METHOD.TUNING.HIGH_FREQUENCY_RATIO_TEMPORAL_RECONSTRUCTION)
        T, peaks_max = compute_period(SS_filtered=SS_filtered,
                                      min_peaks_distance=config.TEMPORAL_METHOD.TUNING.MIN_PEAKS_DISTANCE_PERIOD)
        if config.TEMPORAL_METHOD.DEBUG_MODE:
            draw_results(Im, angle, corr_car, radon_matrix, variance, sinogram_max_var, sinogram_tuned, argmax,
                         wave_length_peaks, wave_length, config, celerity, peaks_max, SS_filtered, T,
                         config.TEMPORAL_METHOD.DEBUG_PATH)

        return {'cel': np.array([celerity]),
                'nu': np.array([1 / wave_length]),
                'T': np.array([T]),
                'dir': np.array([angle]),
                'dcel': np.array([0])
                }
    except Exception:
        print("Bathymetry computation failed")


def spatial_correlation_method(Im, config):
    """
        Bathymetry computation function based on spatial correlation

        Parameters
        ----------
        Im : numpy.ndarray
            Sub-windowed images in M x N x BANDS
         Returns
        -------
        dict:
            As output we deliver a dictionary containing
                -   cel     =   Wave celerity               [m/s]
                -   nu       =   linear Wave number                 [1/m]
                -   L       =   Wavelength                  [m]
                -   T       =   Approximate wave period     [sec]
                -   dir     =   Wave direction (RADON)      [degrees]

        """
    try:
        if config.TEMPORAL_METHOD.PASS_BAND_FILTER:
            Im, flag = fft_filtering(Im, config.TEMPORAL_METHOD.RESOLUTION.SPATIAL,
                                     T_max=config.PREPROCESSING.PASSBAND.HIGH_PERIOD,
                                     T_min=config.PREPROCESSING.PASSBAND.LOW_PERIOD)
        simg_filtered, xx, yy = create_sequence_time_series_spatial(Im=Im)
        angles, distances = compute_angles_distances(M=simg_filtered)
        corr = compute_spatial_correlation(sequence_thumbnail=simg_filtered,
                                           number_frame_shift=config.SPATIAL_METHOD.TEMPORAL_LAG)
        corr_tuned = correlation_tuning(correlation_matrix=corr,
                                        ratio=config.SPATIAL_METHOD.TUNING.RATIO_SIZE_CORRELATION)
        (sinogram_max_var, angle, variance, radon_matrix) = compute_sinogram(correlation_matrix=corr_tuned,
                                                                             median_filter_kernel_ratio=config.SPATIAL_METHOD.TUNING.MEDIAN_FILTER_KERNEL_RATIO_SINOGRAM,
                                                                             mean_filter_kernel_size=config.SPATIAL_METHOD.TUNING.MEAN_FILTER_KERNEL_SIZE_SINOGRAM)
        sinogram_tuned = sinogram_tuning(sinogram=sinogram_max_var,
                                         mean_filter_kernel_size=config.SPATIAL_METHOD.TUNING.MEAN_FILTER_KERNEL_SIZE_SINOGRAM)
        wave_length, zeros = compute_wave_length(sinogram=sinogram_tuned)
        celerity, argmax = compute_celerity(sinogram=sinogram_tuned, wave_length=wave_length,
                                            spatial_resolution=config.SPATIAL_METHOD.RESOLUTION.SPATIAL,
                                            time_resolution=config.SPATIAL_METHOD.RESOLUTION.TEMPORAL,
                                            temporal_lag=config.SPATIAL_METHOD.TEMPORAL_LAG)
        SS = temporal_reconstruction(angle=angle, angles=np.degrees(angles), distances=distances, celerity=celerity,
                                     correlation_matrix=corr,
                                     time_interpolation_resolution=config.SPATIAL_METHOD.RESOLUTION.TIME_INTERPOLATION)
        SS_filtered = temporal_reconstruction_tuning(SS,
                                                     time_interpolation_resolution=config.SPATIAL_METHOD.RESOLUTION.TIME_INTERPOLATION,
                                                     low_frequency_ratio=config.SPATIAL_METHOD.TUNING.LOW_FREQUENCY_RATIO_TEMPORAL_RECONSTRUCTION,
                                                     high_frequency_ratio=config.SPATIAL_METHOD.TUNING.HIGH_FREQUENCY_RATIO_TEMPORAL_RECONSTRUCTION)
        T, peaks_max = compute_period(SS_filtered=SS_filtered,
                                      min_peaks_distance=config.SPATIAL_METHOD.TUNING.MIN_PEAKS_DISTANCE_PERIOD)

        return {'cel': np.array([celerity]),
                'nu': np.array([1 / wave_length]),
                'T': np.array([T]),
                'dir': np.array([angle]),
                'dcel': np.array([0])
                }
    except Exception:
        print("Bathymetry computation failed")


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
