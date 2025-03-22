# -*- coding: utf-8 -*-
""" Exceptions used in bathymetry estimation

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 20 mai 2021

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
from typing import Optional


class WavesException(Exception):
    """ Base class for all waves estimation exceptions
    """

    def __init__(self, reason: Optional[str] = None) -> None:
        super().__init__()
        self.reason = reason

    def __str__(self) -> str:
        if self.reason is None:
            return ''
        return f'{self.reason}'


class WavesEstimationError(WavesException):
    """ Exception raised when an error occurs in bathymetry estimation
    """


class SequenceImagesError(WavesException):
    """ Exception raised when sequence images can not be properly exploited
    """


class NotExploitableSinogram(WavesException):
    """ Exception raised when sinogram can not be exploited
    """


class CorrelationComputationError(WavesException):
    """ Exception raised when correlation can not be computed
    """


class DebugDisplayError(WavesException):
    """ Exception raised when debug display fails
    """


class ProductNotFound(WavesException):
    """ Exception raised when a product cannot be found
    """


class WavesIndexingError(WavesException):
    """ Exception raised when a point cannot be found in the sampling with its coordinates.
    """


class WavesEstimationAttributeError(WavesException):
    """ Exception raised when an attribute is not available in an estimation.
    """
