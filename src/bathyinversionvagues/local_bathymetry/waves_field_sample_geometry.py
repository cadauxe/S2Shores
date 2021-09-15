# -*- coding: utf-8 -*-
""" Class handling the information describing a waves field sample..

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 6 mars 2021
"""
import numpy as np


class WavesFieldSampleGeometry:
    """ This class encapsulates the geometric information defining a sample of a wave field:

    - its direction relative to some origin direction (image or geographical azimuth),
    - its wavelength, considering that a wave field is modeled by a periodic pattern

    This information is strictly local (thus the term sample) and contains only elements which are
    observable. For instance no information related to the dynamics or the bathymetry or anything
    else is contained herein.
    """

    def __init__(self) -> None:
        self._direction = np.nan
        self._wavelength = np.nan

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
        result = f'direction: {self.direction}°\n'
        result += f'wavelength: {self.wavelength:5.2f} (m) wavenumber: {self.wavenumber:8.6f} (m-1)'
        return result
