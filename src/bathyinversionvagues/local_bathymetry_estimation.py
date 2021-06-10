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
from .numpy_utils import sc_all
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
    bathy : dictionary
        This dictionary contains settings and image specifics.
    theta : numpy.ndarray
        M x 1 array with angles to execute the Radon transform over.
    kfft : numpy.ndarray
        M x 1 array with LINEAR wave number.
    phi_min : numpy.ndarray
        M x N of size(kfft,theta) that contains the minimum phase shift based on shallow water.
    phi_deep : numpy.ndarray
        M x N of size(kfft,theta) that contains the maximum phase shift based on deep water.

     Returns
    -------
    dict:
        As output we deliver a dictionary containing
            -   cel     =   Wave celerity               [m/s]
            -   k       =   Wave number                 [1/m]
            -   L       =   Wavelength                  [m]
            -   T       =   Approximate wave period     [sec]
            -   dir     =   Wave direction (RADON)      [degrees]
            -   coh     =   Coherence squared           [-]
            -   dPhi    =   Measured phase shift        [rad]
            -   depth   =   Estimated depth             [m]
    """
    config = estimator.waveparams
    resolution = estimator.waveparams.DX  # in meter
    # Check if the image is NOT empty (if statement):
    if sc_all(Im):

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
                                                            config.MIN_T, config.MAX_T)
        metrics = local_bathy_estimator.metrics

    return results, metrics


def wave_parameters_and_bathy_estimation(sequence, config, delta_t_arrays=None, estimator=None):
    wave_bathy_point = None

    # calcul des paramètres des vagues
    if config.WAVE_EST_METHOD == "TEMPORAL_CORRELATION":
        wave_point = temporal_correlation_method(sequence, config)
    elif config.WAVE_EST_METHOD == "SPATIAL_CORRELATION":
        wave_point = spatial_correlation_method(sequence, config)
    elif config.WAVE_EST_METHOD == "SPATIAL_DFT":
        wave_point, wave_metrics = spatial_dft_estimator(sequence, estimator)
    else:
        raise NotImplementedError

    # inversion de la bathy à partir des paramètres des vagues
    if config.DEPTH_EST_METHOD == "LINEAR":
        wave_bathy_point = depth_linear_inversion(wave_point, config)
    else:
        raise NotImplementedError

    return wave_bathy_point
