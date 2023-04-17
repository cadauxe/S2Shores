# -*- coding: utf-8 -*-
""" Class for debugging the Spatial Correlation estimator.

:author: Yannick Lasne
:organization: THALES c/o CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 28 novembre 2022
"""
from matplotlib import pyplot as plt

from ..local_bathymetry.spatial_correlation_bathy_estimator import \
    SpatialCorrelationBathyEstimator
from .local_bathy_estimator_debug import LocalBathyEstimatorDebug
from .wave_fields_display import (display_sinograms_1D_analysis_spatial_correlation,
                                  display_sinograms_spatial_correlation,
                                  display_waves_images_spatial_correl)


class SpatialCorrelationBathyEstimatorDebug(
        LocalBathyEstimatorDebug, SpatialCorrelationBathyEstimator):
    """ Class allowing to debug the estimations made by a SpatialCorrelationBathyEstimation
    """

    def explore_results(self) -> None:

        print(f'estimations after direction refinement :')
        print(self.bathymetry_estimations)

        # Displays
        plt.show()
        waves_image = display_waves_images_spatial_correl(self)
        dft_sinograms = display_sinograms_spatial_correlation(self)
        dft_sino_spectral = display_sinograms_1D_analysis_spatial_correlation(self)
        waves_image.show()
        dft_sinograms.show()
        dft_sino_spectral.show()
