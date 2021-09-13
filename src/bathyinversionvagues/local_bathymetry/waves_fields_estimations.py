# -*- coding: utf-8 -*-
""" Class handling the information describing the estimations done on a single location.

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 11 sep 2021
"""
import numpy as np


# TODO: add logics for handling dimensions?
class WavesFieldsEstimations(list):
    """ This class gathers information relevant to some location, whatever the bathymetry estimators,
    as well as a list of bathymetry estimations made at this location.
    """

    def __init__(self) -> None:
        super().__init__()

        self._distance_to_shore = np.nan
        self._gravity = np.nan
        self._data_available = True
        self._location = None

    @property
    def distance_to_shore(self) -> float:
        """ :returns: The distance from this point to the nearest shore (km)"""
        return self._distance_to_shore

    @distance_to_shore.setter
    def distance_to_shore(self, value: float) -> None:
        self._distance_to_shore = value
