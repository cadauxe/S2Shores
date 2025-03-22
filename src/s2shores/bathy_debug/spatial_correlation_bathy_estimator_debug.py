# -*- coding: utf-8 -*-
""" Class for debugging the Spatial Correlation estimator.

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 28 novembre 2022

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
from matplotlib import pyplot as plt

from ..local_bathymetry.spatial_correlation_bathy_estimator import SpatialCorrelationBathyEstimator
from .local_bathy_estimator_debug import LocalBathyEstimatorDebug
from .spatial_correlation_wave_fields_display import (display_sinograms_1D_analysis_spatial_correlation,
                                  display_sinograms_spatial_correlation,
                                  display_waves_images_spatial_correl)


class SpatialCorrelationBathyEstimatorDebug(
        LocalBathyEstimatorDebug, SpatialCorrelationBathyEstimator):
    """ Class allowing to debug the estimations made by a SpatialCorrelationBathyEstimation
    """

    def explore_results(self) -> None:

        print('estimations after direction refinement :')
        print(self.bathymetry_estimations)

        # Displays
        if len(self.bathymetry_estimations) > 0:
            waves_image = display_waves_images_spatial_correl(self)
            spatial_correl_sinograms = display_sinograms_spatial_correlation(self)
            spatial_correl_sino_analysis = display_sinograms_1D_analysis_spatial_correlation(self)
            waves_image.show()
            spatial_correl_sinograms.show()
            spatial_correl_sino_analysis.show()
            plt.show()
        else:
            print('No estimation to display.')
