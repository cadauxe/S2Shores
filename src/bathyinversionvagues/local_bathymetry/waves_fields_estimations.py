# -*- coding: utf-8 -*-
""" Class handling the information describing the estimations done on a single location.

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 11 sep 2021
"""
from enum import IntEnum
from typing import Union, List

import numpy as np

from ..image.image_geometry_types import PointType
from ..waves_exceptions import WavesEstimationAttributeError


class SampleStatus(IntEnum):
    """ Enum specifying the synthetic status which can be given to a point in the product."""
    SUCCESS = 0
    FAIL = 1
    ON_GROUND = 2
    NO_DATA = 3
    NO_DELTA_TIME = 4
    OUTSIDE_ROI = 5


# TODO: add logics for handling dimensions?
class WavesFieldsEstimations(list):
    """ This class gathers information relevant to some location, whatever the bathymetry
    estimators, as well as a list of bathymetry estimations made at this location.
    """

    def __init__(self, location: PointType, gravity: float,
                 distance_to_shore: float, inside_roi: bool) -> None:
        super().__init__()

        self._location = location
        self._gravity = gravity
        self._distance_to_shore = distance_to_shore
        self._inside_roi = inside_roi

        self._data_available = True
        self._delta_time_available = True

    def get_property(self, property_name: str) -> Union[float, List[float]]:
        """ Retrieve the values of a property either at the level of WavesFieldsEstimations or
        in the list of WavesFieldEstimation

        :param property_name: name of the estimation property to retrieve
        :returns: the values of the property either as a scalar or a list of values
        :raises WavesEstimationAttributeError: when the property does not exist
        """
        # Firstly try to find the property from the estimations common properties
        if hasattr(self, property_name):
            # retrieve property from the estimations header
            waves_field_property = getattr(self, property_name)
        else:
            if len(self) == 0:
                err_msg = f'Attribute {property_name} undefined (no estimations)'
                raise WavesEstimationAttributeError(err_msg)
            # retrieve property in the list of estimations
            waves_field_property = []
            try:
                for waves_field_estimation in self:
                    waves_field_property.append(getattr(waves_field_estimation, property_name))
            except AttributeError:
                err_msg = f'Attribute {property_name} undefined for {type(waves_field_estimation)}'
                raise WavesEstimationAttributeError(err_msg)

        return waves_field_property

    @property
    def location(self) -> PointType:
        """ :returns: The (X, Y) coordinates of this estimation location"""
        return self._location

    @property
    def distance_to_shore(self) -> float:
        """ :returns: The distance from this estimation location to the nearest shore (km)"""
        return self._distance_to_shore

    @property
    def inside_roi(self) -> bool:
        """ :returns: True if the point is inside the defined ROI, False otherwise"""
        return self._inside_roi

    @property
    def gravity(self) -> float:
        """ :returns: the acceleration of the gravity at this estimation location (m/s2)
        """
        return self._gravity

    @property
    def data_available(self) -> bool:
        """ :returns: True if data was available for doing the estimations, False otherwise """
        return self._data_available

    @data_available.setter
    def data_available(self, value: bool) -> None:
        self._data_available = value

    @property
    def delta_time_available(self) -> bool:
        """ :returns: True if delta time was available for doing estimations, False otherwise """
        return self._delta_time_available

    @delta_time_available.setter
    def delta_time_available(self, value: bool) -> None:
        self._delta_time_available = value

    @property
    def success(self) -> bool:
        """ :returns: True if estimations were run successfully, False otherwise """
        return len(self) > 0

    @property
    def sample_status(self) -> int:
        """ :returns: a synthetic value giving the final estimation status
        """
        status = SampleStatus.SUCCESS
        if self.distance_to_shore <= 0.:
            status = SampleStatus.ON_GROUND
        elif not self.inside_roi:
            status = SampleStatus.OUTSIDE_ROI
        elif not self.data_available:
            status = SampleStatus.NO_DATA
        elif not self.delta_time_available:
            status = SampleStatus.NO_DELTA_TIME
        elif not self.success:
            status = SampleStatus.FAIL
        return status.value

    def __str__(self) -> str:
        result = f'+++++++++ Set of estimations made at: {self.location} \n'
        result += f'  distance to shore: {self.distance_to_shore}   gravity: {self.gravity}\n'
        result += f'  availability: '
        result += f' (data: {self.data_available}, delta time: {self.delta_time_available})\n'
        result += f'  STATUS: {self.sample_status}'
        result += f' (0: SUCCESS, 1: FAIL, 2: ON_GROUND, 3: NO_DATA, 4: NO_DELTA_TIME,'
        result += f' 5: OUTSIDE_ROI)\n'
        result += f'{len(self)} estimations available:\n'
        for index, estimation in enumerate(self):
            result += f'---- estimation {index} ---- type: {type(estimation).__name__}\n'
            result += str(estimation) + '\n'
        return result
