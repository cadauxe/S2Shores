# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:45:01 2020

This is a bathymetry inversion package with all kinds of functions for
depth inversion. Initially designed for TODO

@author: erwinbergsma
         gregoirethoumyre
"""
from typing import Optional, List  # @NoMove

import warnings

import numpy as np

from ..depthinversionmethods import depth_linear_inversion
from ..image_processing.waves_image import WavesImage
from ..waves_exceptions import WavesException

from .spatial_dft_bathy_estimator import SpatialDFTBathyEstimator
from .wavemethods import spatial_correlation_method
from .wavemethods import temporal_correlation_method


def spatial_dft_estimator(images_sequence: List[WavesImage], global_estimator,
                          selected_directions: Optional[np.ndarray]=None):
    """
    """
    local_bathy_estimator = SpatialDFTBathyEstimator(images_sequence,
                                                     global_estimator,
                                                     selected_directions=selected_directions)

    try:
        local_bathy_estimator.run()
    except WavesException as excp:
        warnings.warn(f'Unable to estimate bathymetry: {str(excp)}')

    results = local_bathy_estimator.get_results_as_dict()
    metrics = local_bathy_estimator.metrics

    # TODO: replace dictionaries by local_bathy_estimator object return when other estimator
    # are updated.
    return results, metrics


def temporal_correlation_estimator(images_sequence: List[WavesImage], global_estimator,
                                   selected_directions: Optional[np.ndarray]=None):
    """
    """
    wave_point = temporal_correlation_method(images_sequence, global_estimator)
    # inversion de la bathy à partir des paramètres des vagues
    if global_estimator.waveparams.DEPTH_EST_METHOD == 'LINEAR':
        wave_bathy_point = depth_linear_inversion(wave_point, global_estimator)
    else:
        msg = f'{global_estimator.waveparams.DEPTH_EST_METHOD} '
        msg += 'is not a supported depth estimation method.'
        raise NotImplementedError(msg)

    results = wave_bathy_point
    metrics = {}
    # TODO: replace dictionaries by local_bathy_estimator object return when this estimator
    # is updated.
    return results, metrics


def spatial_correlation_estimator(images_sequence: List[WavesImage], global_estimator,
                                  selected_directions: Optional[np.ndarray]=None):
    """
    """
    wave_point = spatial_correlation_method(images_sequence, global_estimator)
    # inversion de la bathy à partir des paramètres des vagues
    if global_estimator.waveparams.DEPTH_EST_METHOD == 'LINEAR':
        wave_bathy_point = depth_linear_inversion(wave_point, global_estimator)
    else:
        msg = f'{global_estimator.waveparams.DEPTH_EST_METHOD} '
        msg += 'is not a supported depth estimation method.'
        raise NotImplementedError(msg)

    results = wave_bathy_point
    metrics = {}
    # TODO: replace dictionaries by local_bathy_estimator object return when this estimator
    # is updated.
    return results, metrics


def wave_parameters_and_bathy_estimation(images_sequence: List[WavesImage],
                                         global_estimator, delta_t_arrays=None):

    wave_bathy_point = None

    # calcul des paramètres des vagues
    if global_estimator.waveparams.WAVE_EST_METHOD == 'SPATIAL_DFT':

        wave_bathy_point, wave_metrics = spatial_dft_estimator(images_sequence,
                                                               global_estimator)
    elif global_estimator.waveparams.WAVE_EST_METHOD == 'TEMPORAL_CORRELATION':
        wave_bathy_point, wave_metrics = temporal_correlation_estimator(images_sequence,
                                                                        global_estimator)
    elif global_estimator.waveparams.WAVE_EST_METHOD == 'SPATIAL_CORRELATION':
        wave_bathy_point, wave_metrics = spatial_correlation_estimator(images_sequence,
                                                                       global_estimator)
    else:
        msg = f'{global_estimator.waveparams.WAVE_EST_METHOD} is not a supported '
        msg += 'local bathymetry estimation method.'
        raise NotImplementedError(msg)

    return wave_bathy_point
