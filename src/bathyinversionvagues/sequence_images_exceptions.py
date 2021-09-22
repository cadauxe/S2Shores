# -*- coding: utf-8 -*-
""" Exceptions used in bathymetry estimation

:author: DEGOUL Romain
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 22 septembre 2021
"""
from typing import Optional


class SequenceImagesException(Exception):
    """ Base class for all sequence images exceptions
    """

    def __init__(self, reason: Optional[str] = None) -> None:
        super().__init__()
        self.reason = reason

    def __str__(self) -> str:
        if self.reason is None:
            return ''
        return f'{self.reason}'


class SequenceImagesEmptyError(SequenceImagesException):
    """ Exception raised when sequence images is empty
    """


class SequenceImagesSizeError(SequenceImagesException):
    """ Exception raised when sequence images do not have same size
    """


class SequenceImagesResolutionError(SequenceImagesException):
    """ Exception raised when sequence images do not have same resolution
    """
