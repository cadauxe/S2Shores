# -*- coding: utf-8 -*-
""" Class handling the information describing the estimations done on a single location.

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 11 sep 2021
"""
from enum import IntEnum

import numpy as np


class SampleStatus(IntEnum):
    SUCCESS = 0
    FAIL = 1
    ON_GROUND = 2
    NO_DATA = 3

# TODO: add logics for handling dimensions?


class WavesFieldsEstimations(list):
    """ This class gathers information relevant to some location, whatever the bathymetry
    estimators, as well as a list of bathymetry estimations made at this location.
    """

    def __init__(self) -> None:
        super().__init__()

        self._distance_to_shore = np.nan
        self._gravity = np.nan
        self._data_available = True
        self._location = None
        self._success = True

    @property
    def distance_to_shore(self) -> float:
        """ :returns: The distance from this estimation location to the nearest shore (km)"""
        return self._distance_to_shore

    @distance_to_shore.setter
    def distance_to_shore(self, value: float) -> None:
        self._distance_to_shore = value

    @property
    def gravity(self) -> float:
        """ :returns: the acceleration of the gravity at this estimation location (m/s2)
        """
        return self._gravity

    @gravity.setter
    def gravity(self, value: float) -> None:
        self._gravity = value

    @property
    def data_available(self) -> bool:
        """ :returns: True if data was available for doing the estimations, False otherwise """
        return self._data_available

    @data_available.setter
    def data_available(self, value: bool) -> None:
        self._data_available = value

    @property
    def success(self) -> bool:
        """ :returns: True if estimations were run successfully, False otherwise """
        return self._success

    @success.setter
    def success(self, value: bool) -> None:
        self._success = value

    @property
    def sample_status(self) -> int:
        """ :returns: a synthetic value giving the final estimation status
        """
        status = SampleStatus.SUCCESS
        if self.distance_to_shore == 0.:
            status = SampleStatus.ON_GROUND
        elif not self.data_available:
            status = SampleStatus.NO_DATA
        elif not self.success:
            status = SampleStatus.FAIL
        return status.value
