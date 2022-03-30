# -*- coding: utf-8 -*-
""" Class handling the information describing the estimations done on a single location.

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 11 sep 2021
"""
from enum import IntEnum
import warnings

from typing import Union, List, Optional

import numpy as np

from ..image.image_geometry_types import PointType
from ..waves_exceptions import WavesEstimationAttributeError

from .waves_field_estimation import WavesFieldEstimation


class SampleStatus(IntEnum):
    """ Enum specifying the synthetic status which can be given to a point in the product."""
    SUCCESS = 0
    FAIL = 1
    ON_GROUND = 2
    NO_DATA = 3
    NO_DELTA_TIME = 4
    OUTSIDE_ROI = 5


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

    def append(self, estimation: WavesFieldEstimation) -> None:
        """ Store a single estimation into the estimations list, ensuring that there are no
        duplicate estimations for the same (direction, wavelength) pair.

        :param estimation: a new estimation to store inside this localized list of estimations
        """
        stored_wavelengths_directions = [(estimation.wavelength, estimation.direction)
                                         for estimation in self]
        # Do not store duplicate estimations for the same direction/wavelength
        if (estimation.wavelength, estimation.direction) in stored_wavelengths_directions:
            warnings.warn(f'Trying to store a duplicate estimation:\n {str(estimation)} ')
            return
        super().append(estimation)

    def sort_on_attribute(self, attribute_name: Optional[str] = None, reverse: bool = True) -> None:
        """ Sort in place the waves fields estimations based on one of their attributes.

        :param attribute_name: name of an attribute present in all estimations to use for sorting
        :param reverse: When True sorting is in descending order, when False in ascending order
        """
        if attribute_name is not None:
            self.sort(key=lambda x: getattr(x, attribute_name), reverse=reverse)

    def argsort_on_attribute(self, attribute_name: Optional[str] = None,
                             reverse: bool = True) -> List[int]:
        """ Return the indices of the waves fields estimations which would sort them based
        on one of their attributes.

        :param attribute_name: name of an attribute present in all estimations to use for sorting
        :param reverse: When True sorting is in descending order, when False in ascending order
        :returns: either en empty list if attribute_name is None or the list of indices which would
                  sort this WavesFieldsEstimations according to one of the attributes.
        """
        if attribute_name is not None:
            attr_list = [getattr(estimation, attribute_name) for estimation in self]
            arg_sorted = np.argsort(attr_list).tolist()
            if reverse:
                arg_sorted.reverse()
            return arg_sorted
        return []

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
            if not self:
                err_msg = f'Attribute {property_name} undefined (no estimations)'
                raise WavesEstimationAttributeError(err_msg)
            waves_field_property = self.get_estimations_attribute(property_name)
        return waves_field_property

    def get_estimations_attribute(self, attribute_name: str) -> List[float]:
        """ Retrieve the values of some attribute in the list of stored waves field estimations.

        :param attribute_name: name of the attribute to retrieve
        :returns: the values of the attribute in the order where the estimations are stored
        :raises WavesEstimationAttributeError: when the property does not exist in at least
                                               one estimation
        """
        try:
            return [getattr(estimation, attribute_name) for estimation in self]
        except AttributeError:
            err_msg = f'Attribute {attribute_name} undefined for some waves field estimation'
            raise WavesEstimationAttributeError(err_msg)

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
