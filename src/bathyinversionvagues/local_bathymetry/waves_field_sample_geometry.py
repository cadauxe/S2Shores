# -*- coding: utf-8 -*-
""" Class handling the information describing a waves field sample..

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 6 mars 2021
"""
from typing import Optional, Tuple

import numpy as np


class WavesFieldSampleGeometry:
    """ This class encapsulates the geometric information defining a sample of a wave field:

    - its position either relative to the image or to geographic coordinates,
    - its direction relative to some origin direction (image or geographical azimuth),
    - its wavelength, considering that a wave field is modeled by a periodic pattern

    This information is strictly local (thus the term sample) and contains only elements which are
    observable. For instance no information related to the dynamics or the bathymetry or anything
    else is contained herein.
    """

    def __init__(self) -> None:
        self._position: Optional[Tuple[bool, float, float]] = None
        self._direction = np.nan
        self._wavelength = np.nan

    @property
    def position(self) -> Optional[Tuple[bool, float, float]]:
        """ 3 elements defining the position of the sample:
                         - the first one is a boolean which indicates if the position is defined
                           in image coordinates (True) or geographical coordinates (False).
                         - the second and third ones define the coordinates of the sample in
                           the 2D coordinates system defined by the first element.
        :returns: The position of this sample.
        :raises TypeError: when argument is not a valid 3-tuple or None
        """
        return self._position

    @position.setter
    def position(self, value: Optional[Tuple[bool, float, float]]) -> None:
        if value is not None and (not isinstance(value, tuple) or
                                  len(value) != 3 or
                                  not isinstance(value[0], bool) or
                                  not isinstance(value[1], float) or
                                  not isinstance(value[2], float)):
            msg = 'position attribute accepts only None or a tuple (bool, float, float)'
            raise TypeError(msg)
        self._position = value

    @property
    def direction(self) -> float:
        """ :returns: The waves field direction relative to its position (degrees)"""
        return self._direction

    @direction.setter
    def direction(self, value: float) -> None:
        self._direction = value

    @property
    def wavelength(self) -> float:
        """ :returns: The waves field wavelength (m)"""
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value: float) -> None:
        self._wavelength = value

    @property
    def wavenumber(self) -> float:
        """ :returns: The waves field wave number (m-1)"""
        return 1. / self._wavelength

    @wavenumber.setter
    def wavenumber(self, value: float) -> None:
        self.wavelength = 1. / value

    def __str__(self) -> str:
        result = f'position: {self.position}    direction: {self.direction}°\n'
        result += f'wavelength: {self.wavelength:5.2f} (m) wavenumber: {self.wavenumber:8.6f} (m-1)'
        return result
