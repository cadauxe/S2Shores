# -*- coding: utf-8 -*-
""" Definition of the EstimatedBathy class

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 14/05/2021

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, Hashable, List, Mapping

import numpy as np  # @NoMove
from xarray import DataArray, Dataset  # @NoMove

from ..waves_exceptions import WavesEstimationAttributeError
from .bathymetry_sample_estimations import BathymetrySampleEstimations

DEBUG_LAYER = ['DEBUG']
EXPERT_LAYER = DEBUG_LAYER + ['EXPERT']
NOMINAL_LAYER = EXPERT_LAYER + ['NOMINAL']
ALL_LAYERS_TYPES = NOMINAL_LAYER

METERS_UNIT = 'Meters [m]'
SPATIAL_REF = 'spatial_ref'


class EstimatedBathy():
    """ This class gathers all the estimated bathymetry samples in a whole dataset.
    """

    bathy_product_def: Dict[str, Dict[str, Any]]

    def __init__(self, acq_time: str) -> None:
        """ Define dimensions for which the estimated bathymetry samples will be defined.
        :param acq_time: the time at which the bathymetry samples are estimated
        """
        timestamp = datetime(int(acq_time[:4]), int(acq_time[4:6]), int(acq_time[6:8]),
                             int(acq_time[9:11]), int(acq_time[11:13]), int(acq_time[13:15]))
        self.timestamps = [timestamp]

    @classmethod
    @abstractmethod
    def store_estimations(self, index: int, bathy_estimations: BathymetrySampleEstimations) -> None:
        """ Store a set of bathymetry estimations at some location

        :param index: index where to store the estimations
        :param bathy_estimations: the whole set of bathy estimations data at one point.
        """

    @classmethod
    @abstractmethod
    def _build_data_array(self, sample_property: str,
                          layer_definition: Dict[str, Any], nb_keep: int) -> DataArray:
        """ Build an xarray DataArray containing one estimated bathymetry property.

        :param sample_property: name of the property to format as a DataArray
        :param layer_definition: definition of the way to format the property
        :param nb_keep: the number of different bathymetry estimations to keep for one location.
        :raises IndexError: when the property is not a scalar or a vector
        :returns: an xarray DataArray containing the formatted property
        :raises WavesEstimationAttributeError: when the requested property cannot be found in the
                                               estimations attributes.
        """

    # TODO: split array filling in two methods: one for 2D (X, Y) and one for 3D (X, Y, kKeep)
    @classmethod
    @abstractmethod
    def _fill_array(self, sample_property: str, layer_data: np.ndarray, index: List[int]) -> None:
        """ Fill the layer_data array at a given index (1D: points, 2D: (X, Y))
        """

    @classmethod
    @abstractmethod
    def _get_coords(self, dims: List[str], nb_keep: int) -> Mapping[Hashable, Any]:
        """ Get coordinates dictionary
        :param dims:
        :param nb_keep:
        :raise ValueError: if unknown dimension used in dims
        :return dict_coords: dictionary with coordinates
        """

    def build_dataset(self, layers_type: str, nb_keep: int) -> Dataset:
        """ Build an xarray DataSet containing the estimated bathymetry.

        :param layers_type: select the layers which will be produced in the dataset.
                            Value must be one of ALL_LAYERS_TYPES.
        :param nb_keep: the number of different bathymetry estimations to keep for one location.
        :raises ValueError: when layers_type is not equal to one of the accepted values
        :returns: an xarray Dataset containing the estimated bathymetry.
        """
        if layers_type not in ALL_LAYERS_TYPES:
            msg = f'incorrect layers_type ({layers_type}). Must be one of: {ALL_LAYERS_TYPES}'
            raise ValueError(msg)

        data_arrays = {}

        # build individual DataArray with attributes:
        for sample_property, layer_definition in self.bathy_product_def.items():
            if layers_type in layer_definition['layer_type']:
                try:
                    data_array = self._build_data_array(sample_property, layer_definition, nb_keep)
                    data_arrays[layer_definition['layer_name']] = data_array
                except WavesEstimationAttributeError:
                    # property was not found at any location: ignore it
                    continue

        # Combine all DataArray in a single Dataset:
        return Dataset(data_vars=data_arrays)
