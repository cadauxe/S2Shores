from bathycommun.config.config_bathy import ConfigBathy
from bathyinversionvagues.wavemethods import spatial_dft_method
from bathyinversionvagues.depthinversionmethods import depth_linear_inversion


def wave_parameters_and_bathy_estimation(sequence, delta_t_arrays=None, k_fft=None, phi_min=None, phi_deep=None):
    wave_bathy_point = None
    config = ConfigBathy()

    # calcul des paramètres des vagues
    if config.wave_estimation_method == "TEMPORAL_CORRELATION":
        pass
    elif config.wave_estimation_method == "SPATIAL_CORRELATION":
        pass
    elif config.wave_estimation_method == "SPATIAL_DFT":
        wave_point = spatial_dft_method(sequence, config, k_fft, phi_min, phi_deep)
    else:
        raise NotImplementedError

    # inversion de la bathy à partir des paramètres des vagues
    if config.depth_estimation_method == "LINEAR":
        wave_bathy_point = depth_linear_inversion(wave_point, config)
    else:
        raise NotImplementedError

    return wave_bathy_point
