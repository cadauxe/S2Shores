# -*- coding: utf-8 -*-
""" Definition of the BathyEstimatorParameters class

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 17/05/2021

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
from typing import Optional, Tuple

from ..image.ortho_stack import FramesIdsType


class BathyEstimatorParameters:
    """ Definition of global bathy estimator properties, mapping Munchified parameters.
    """

    def __init__(self, wave_params: dict) -> None:
        """ Constructor

        :param wave_params: parameters for the global and local bathymetry estimators
        """
        self._waveparams = wave_params

    @property
    def _global_estimator_params(self) -> dict:
        """ :returns: the set of parameters of the global estimator
        """
        return self._waveparams['GLOBAL_ESTIMATOR']

    @property
    def output_format(self) -> str:
        """ :returns: the output format to use
        """
        return self._global_estimator_params['OUTPUT_FORMAT']

    @property
    def chains_versions(self) -> str:
        """ :returns: the versions of the packages used in the chain
        """
        return self._waveparams['CHAINS_VERSIONS']

    @property
    def local_estimator_code(self) -> str:
        """ :returns: the code of the local estimator to use with this global estimator
        """
        return self._global_estimator_params['WAVE_EST_METHOD']

    @property
    def selected_frames_param(self) -> Optional[FramesIdsType]:
        """ :returns: the list of frames selected for running the estimation.
        """
        try:
            result = self._global_estimator_params['SELECTED_FRAMES']
        except AttributeError:
            result = None
        return result

    @property
    def nb_max_wave_fields(self) -> int:
        """ :returns: the maximum number of wave fields to keep
        """
        return self._global_estimator_params['NKEEP']

    @property
    def layers_type(self) -> str:
        """ :returns: the type of layers to write in the bathymetry product
        """
        return self._global_estimator_params['LAYERS_TYPE']

    @property
    def depth_min(self) -> float:
        """ :returns: the minimum depth (m) to consider when doing inversion
        """
        return self._global_estimator_params['MIN_D']

    @property
    def waves_period_min(self) -> float:
        """ :returns: the minimum waves period (s) to consider when doing inversion
        """
        return self._global_estimator_params['MIN_T']

    @property
    def waves_period_max(self) -> float:
        """ :returns: the maximum waves period (s) to consider when doing inversion
        """
        return self._global_estimator_params['MAX_T']

    @property
    def waves_period_range(self) -> Tuple[float, float]:
        """ :returns: the range of waves period (s) to consider as physical
        """
        return self.waves_period_min, self.waves_period_max

    @property
    def waves_linearity_min(self) -> float:
        """ :returns: the minimum value of waves linearity to consider when doing inversion
        """
        return self._global_estimator_params['MIN_WAVES_LINEARITY']

    @property
    def waves_linearity_max(self) -> float:
        """ :returns: the maximum value of waves linearity to consider when doing inversion
        """
        return self._global_estimator_params['MAX_WAVES_LINEARITY']

    @property
    def waves_linearity_range(self) -> Tuple[float, float]:
        """ :returns: the range of values for waves linearity to consider as physical
        """
        return self.waves_linearity_min, self.waves_linearity_max

    @property
    def sampling_step_x(self) -> float:
        """ :returns: the sampling step (m) along the X axis to define the samples locations
        """
        return self._global_estimator_params['DXP']

    @property
    def sampling_step_y(self) -> float:
        """ :returns: the sampling step (m) along the Y axis to define the samples locations
        """
        return self._global_estimator_params['DYP']

    @property
    def window_size_x(self) -> float:
        """ :returns: the size of the estimation window along the X axis (m)
        """
        return self._global_estimator_params['WINDOW']

    @property
    def window_size_y(self) -> float:
        """ :returns: the size of the estimation window along the Y axis (m)
        """
        return self._global_estimator_params['WINDOW']

    @property
    def smoothing_columns_size(self) -> int:
        """ :returns: the size of the smoothing filter along columns (pixels)
        """
        return self._global_estimator_params['SM_LENGTH']

    @property
    def smoothing_lines_size(self) -> int:
        """ :returns: the size of the smoothing filter along lines (pixels)
        """
        return self._global_estimator_params['SM_LENGTH']

    @property
    def depth_estimation_method(self) -> str:
        """ :returns: the code of the depth estimation method to use for depth inversion
        """
        return self._global_estimator_params['DEPTH_EST_METHOD']

    @property
    def max_offshore_distance(self) -> int:
        """ :returns: Maximum allowed offshore distance from this estimation location (km)
        """
        return self._global_estimator_params['OFFSHORE_LIMIT']

    @property
    def local_estimator_params(self) -> dict:
        """ :returns: the set of parameters specific to the currently defined local estimator
        """
        return self._waveparams[self.local_estimator_code]
