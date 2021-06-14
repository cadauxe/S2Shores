# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:45:01 2020

This is a bathymetry inversion package with all kinds of functions for
depth inversion. Initially designed for TODO

@author: erwinbergsma
         gregoirethoumyre
"""
from typing import Optional  # @NoMove

import warnings

import numpy as np

from .depthinversionmethods import depth_linear_inversion
from .spatial_dft_bathy_estimator import SpatialDFTBathyEstimator
from .wavemethods import spatial_correlation_method
from .wavemethods import temporal_correlation_method
from .waves_exceptions import WavesException
from .waves_image import WavesImage


def spatial_dft_estimator(Im, estimator, selected_directions: Optional[np.ndarray]=None):
    """
    Parameters
    ----------
    Im : numpy.ndarray
        Sub-windowed images in M x N x BANDS -- currently only 2 bands are used!

    """
    config = estimator.waveparams
    resolution = estimator.waveparams.DX  # in meter
    # Check if the image is NOT empty (if statement):

    # Create waves fields estimator [we can make this a for loop for the number of frames]
    # TODO: Link WavesImage to OrthoImage and use resolution from it
    if estimator.smoothing_requested:
        smoothing = (estimator.smoothing_lines_size, estimator.smoothing_columns_size)
    else:
        smoothing = None

    waves_image_ref = WavesImage(Im[:, :, 0], resolution, smoothing=smoothing)
    waves_image_sec = WavesImage(Im[:, :, 1], resolution, smoothing=smoothing)

    local_bathy_estimator = SpatialDFTBathyEstimator(waves_image_ref, waves_image_sec,
                                                     estimator,
                                                     selected_directions=selected_directions)

    try:
        local_bathy_estimator.run()
    except WavesException as excp:
        warnings.warn(f'Unable to estimate bathymetry: {str(excp)}')

    results = local_bathy_estimator.get_results_as_dict(config.NKEEP,
                                                        config.MIN_T,
                                                        config.MAX_T,
                                                        config.MIN_WAVES_LINEARITY,
                                                        config.MAX_WAVES_LINEARITY)
    metrics = local_bathy_estimator.metrics

    # TODO: replace dictionaries by local_bathy_estimator object return when other estimator
    # are updated.
    return results, metrics


def wave_parameters_and_bathy_estimation(sequence, global_estimator, delta_t_arrays=None):

    wave_bathy_point = None

    # calcul des paramètres des vagues
    if global_estimator.waveparams.WAVE_EST_METHOD == "SPATIAL_DFT":
        wave_bathy_point, wave_metrics = spatial_dft_estimator(sequence, global_estimator)
    elif global_estimator.waveparams.WAVE_EST_METHOD == "TEMPORAL_CORRELATION":
        wave_point = temporal_correlation_method(sequence, global_estimator.waveparams)
        # inversion de la bathy à partir des paramètres des vagues
        if global_estimator.waveparams.DEPTH_EST_METHOD == "LINEAR":
            wave_bathy_point = depth_linear_inversion(wave_point, global_estimator.waveparams)
        else:
            msg = f'{global_estimator.waveparams.DEPTH_EST_METHOD} '
            msg += 'is not a supported depth estimation method.'
            raise NotImplementedError(msg)
    elif global_estimator.waveparams.WAVE_EST_METHOD == "SPATIAL_CORRELATION":
        wave_point = spatial_correlation_method(sequence, global_estimator.waveparams)
        # inversion de la bathy à partir des paramètres des vagues
        if global_estimator.waveparams.DEPTH_EST_METHOD == "LINEAR":
            wave_bathy_point = depth_linear_inversion(wave_point, global_estimator.waveparams)
        else:
            msg = f'{global_estimator.waveparams.DEPTH_EST_METHOD} '
            msg += 'is not a supported depth estimation method.'
            raise NotImplementedError(msg)
    else:
        msg = f'{global_estimator.waveparams.WAVE_EST_METHOD} is not a supported '
        msg += 'local bathymetry estimation method.'
        raise NotImplementedError(msg)

    return wave_bathy_point
