# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:45:01 2020

This is a bathymetry inversion package with all kinds of functions for
depth inversion. Initially designed for TODO

@author: erwinbergsma
         gregoirethoumyre
"""
from typing import Optional, List, Callable, Type  # @NoMove

import warnings

import numpy as np

from ..image_processing.waves_image import WavesImage
from ..waves_exceptions import WavesException
from .spatial_correlation_bathy_estimator import spatial_correlation_method
from .spatial_dft_bathy_estimator import SpatialDFTBathyEstimator
from .temporal_correlation_bathy_estimator import temporal_correlation_method


def run_spatial_dft_estimation(images_sequence: List[WavesImage], global_estimator,
                               selected_directions: Optional[np.ndarray]=None):
    return run_local_bathy_estimation(SpatialDFTBathyEstimator, images_sequence, global_estimator,
                                      selected_directions)


def run_temporal_correlation_estimation(images_sequence: List[WavesImage], global_estimator,
                                        selected_directions: Optional[np.ndarray]=None):
    """
    """
    return temporal_correlation_method(images_sequence, global_estimator)


def run_spatial_correlation_estimation(images_sequence: List[WavesImage], global_estimator,
                                       selected_directions: Optional[np.ndarray]=None):
    """
    """
    return spatial_correlation_method(images_sequence, global_estimator)


def run_local_bathy_estimation(local_bathy_estimator_cls: Type,
                               images_sequence: List[WavesImage], global_estimator,
                               selected_directions: Optional[np.ndarray]=None):
    """
    """
    local_bathy_estimator = local_bathy_estimator_cls(images_sequence,
                                                      global_estimator,
                                                      selected_directions=selected_directions)

    try:
        local_bathy_estimator.run()
    except WavesException as excp:
        warnings.warn(f'Unable to estimate bathymetry: {str(excp)}')

    results = local_bathy_estimator.get_results_as_dict()
    # FIXME: decide what to do with metrics
    metrics = local_bathy_estimator.metrics

    # TODO: replace dictionaries by local_bathy_estimator object return when other estimator
    # are updated.
    return results


# FIXME: to be replaced by the classes to be instanciated for local bathy estimators
LOCAL_BATHY_ESTIMATION_FUNC = {'SPATIAL_DFT': run_spatial_dft_estimation,
                               'TEMPORAL_CORRELATION': run_temporal_correlation_estimation,
                               'SPATIAL_CORRELATION': run_spatial_correlation_estimation}

LOCAL_BATHY_ESTIMATION_CLS = {'SPATIAL_DFT': SpatialDFTBathyEstimator,
                              'TEMPORAL_CORRELATION': None,
                              'SPATIAL_CORRELATION': None}


def get_local_bathy_estimator(local_estimator_code: str) -> Callable:
    """ return the local bathymetry estimator function to be called for a given estimator code

    :param local_estimator_code: the code of the function to call
    :returns: the function to call for computing local bathymetry
    """
    try:
        local_bathy_estimator = LOCAL_BATHY_ESTIMATION_FUNC[local_estimator_code]
    except KeyError:
        msg = f'{local_estimator_code} is not a supported local bathymetry estimation method.'
        raise NotImplementedError(msg)
    return local_bathy_estimator
