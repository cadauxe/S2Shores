# -*- coding: utf-8 -*-
""" Exceptions used in bathymetry estimation

:author: GIROS Alain
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 20 mai 2021
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


class NoRadonTransformError(WavesException):
    """ Exception raised when trying to access a non existent Radon transform
    """


class NoDeltaTimeProviderError(WavesException):
    """ Exception raised when using bathymetry estimator without specifying a DeltaTimeProvider
    """
