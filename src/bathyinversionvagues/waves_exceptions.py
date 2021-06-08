# -*- coding: utf-8 -*-
"""
Exceptions used in bathymetry estimation

:author: GIROS Alain
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 20 mai 2021
"""


class WavesException(Exception):
    """ Base class for all waves estimation exceptions
    """


class WavesEstimationError(WavesException):
    """ Exception raised when an error occurs in bathymetry estimation
    """


class NoRadonTransformError(WavesException):
    """ Exception raised when trying to access a non existent Radon transform
    """
