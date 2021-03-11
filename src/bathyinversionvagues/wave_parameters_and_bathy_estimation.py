# -*- coding: ISO-8859-1 -*-

from bathycommun.config.config_bathy import ConfigBathy
from bathyinversionvagues.wavemethods import spatial_dft_method
from bathyinversionvagues.wavemethods import temporal_correlation_method
from bathyinversionvagues.wavemethods import spatial_correlation_method
from bathyinversionvagues.depthinversionmethods import depth_linear_inversion
from pathlib import Path
import os


def wave_parameters_and_bathy_estimation(sequence, delta_t_arrays=None, k_fft=None, phi_min=None, phi_deep=None):
    wave_bathy_point = None
    config = ConfigBathy()

    # calcul des paramètres des vagues
    if config.WAVE_EST_METHOD == "TEMPORAL_CORRELATION":
        wave_point = temporal_correlation_method(sequence,config)
    elif config.WAVE_EST_METHOD == "SPATIAL_CORRELATION":
        wave_point = spatial_correlation_method(sequence,config)
    elif config.WAVE_EST_METHOD == "SPATIAL_DFT":
        wave_point = spatial_dft_method(sequence, config, k_fft, phi_min, phi_deep)
    else:
        raise NotImplementedError

    # inversion de la bathy à partir des paramètres des vagues
    if config.DEPTH_EST_METHOD == "LINEAR":
        wave_bathy_point = depth_linear_inversion(wave_point, config)
    else:
        raise NotImplementedError

    return wave_bathy_point
