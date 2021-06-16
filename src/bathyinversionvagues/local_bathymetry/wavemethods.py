# -*- coding: utf-8 -*-
"""
Created on Wed Feb 3 10:12:00 2021

Module containing all wave parameters estimation methods

@author: erwinbergsma
         gregoirethoumyre
         degoulromain
"""
from typing import List

import numpy as np

from ..image_processing.shoresutils import (fft_filtering, compute_sinogram,
                                            create_sequence_time_series_temporal,
                                            compute_temporal_correlation, compute_celerity,
                                            cartesian_projection, correlation_tuning,
                                            sinogram_tuning, compute_wave_length, compute_period,
                                            temporal_reconstruction,
                                            temporal_reconstruction_tuning,
                                            create_sequence_time_series_spatial,
                                            compute_angles_distances, compute_spatial_correlation)
from ..image_processing.waves_image import WavesImage
from ..waves_fields_display import draw_results


def temporal_correlation_method(images_sequence: List[WavesImage], config):
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
    # FIXME: temporary adaptor before getting rid of stacked np.ndarrays.
    Im = np.dstack([image.pixels for image in images_sequence])

    try:
        if config.TEMPORAL_METHOD.PASS_BAND_FILTER:
            Im, flag = fft_filtering(Im, config.TEMPORAL_METHOD.RESOLUTION.SPATIAL,
                                     config.PREPROCESSING.PASSBAND.HIGH_PERIOD,
                                     config.PREPROCESSING.PASSBAND.LOW_PERIOD, 9.81)
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


def spatial_correlation_method(images_sequence: List[WavesImage], config):
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
    # FIXME: temporary adaptor before getting rid of stacked np.ndarrays.
    Im = np.dstack([image.pixels for image in images_sequence])

    try:
        if config.TEMPORAL_METHOD.PASS_BAND_FILTER:
            Im, flag = fft_filtering(Im, config.TEMPORAL_METHOD.RESOLUTION.SPATIAL,
                                     config.PREPROCESSING.PASSBAND.HIGH_PERIOD,
                                     config.PREPROCESSING.PASSBAND.LOW_PERIOD, 9.81)
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
