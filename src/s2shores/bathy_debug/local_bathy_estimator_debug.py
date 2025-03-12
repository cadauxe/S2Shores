# -*- coding: utf-8 -*-
""" Base class for the estimators of wave fields from several images taken at a small
time interval.

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 5 mars 2021

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
from abc import abstractmethod

from ..local_bathymetry.local_bathy_estimator import LocalBathyEstimator


class LocalBathyEstimatorDebug(LocalBathyEstimator):
    """ Abstract class handling debug mode for LocalBathyEstimator
    """

    def run(self) -> None:
        super().run()
        self.explore_results()

    @abstractmethod
    def explore_results(self) -> None:
        """ Method called when estimator has run to allow results exploration for debugging purposes
        """
