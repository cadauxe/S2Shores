# -*- coding: utf-8 -*-
""" Class for debugging the Spatial Correlation estimator.

:author: Yannick Lasne
:organization: THALES c/o CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 28 novembre 2022
"""

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
        display_waves_images_spatial_correl(self)
        display_sinograms_spatial_correlation(self)
        display_sinograms_1D_analysis_spatial_correlation(self)
