# -*- coding: utf-8 -*-
""" Class handling the information describing a wave field sample..

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2022 CNES. All rights reserved.
:license: see LICENSE file
:created: 29 novembre 2022

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
from ..data_model.bathymetry_sample_estimation import BathymetrySampleEstimation


class SpatialCorrelationBathyEstimation(BathymetrySampleEstimation):
    """ This class encapsulates the information estimated in a wave field sample by a
    SpatialCorrelationBathyEstimator.

    It defines the estimation attributes specific to this estimator.
    """
