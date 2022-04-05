# -*- coding: utf-8 -*-
""" Definition of the EstimatedBathy class

:author: GIROS Alain
:created: 14/05/2021
"""
from datetime import datetime
from typing import Mapping, Hashable, Any, Dict, List

import numpy as np  # @NoMove
from xarray import Dataset, DataArray  # @NoMove


from ..waves_exceptions import WavesEstimationIndexingError, WavesEstimationAttributeError
from .bathymetry_sample_estimations import BathymetrySampleEstimations


DEBUG_LAYER = ['DEBUG']
EXPERT_LAYER = DEBUG_LAYER + ['EXPERT']
NOMINAL_LAYER = EXPERT_LAYER + ['NOMINAL']
ALL_LAYERS_TYPES = NOMINAL_LAYER

DIMS_Y_X_NKEEP_TIME = ['y', 'x', 'kKeep', 'time']
DIMS_Y_X_TIME = ['y', 'x', 'time']


# Provides a mapping from entries into the output dictionary of a local estimator to a netCDF layer.
BATHY_PRODUCT_DEF: Dict[str, Dict[str, Any]] = {
    'status': {'layer_type': NOMINAL_LAYER,
               'layer_name': 'Status',
               'dimensions': DIMS_Y_X_TIME,
               'data_type': np.ushort,
               'fill_value': 0,
               'precision': 0,
               'attrs': {'Dimension': 'Flags',
                         'name': 'Bathymetry estimation status',
                         'comment': '0: SUCCESS, 1: FAIL, 2: ON_GROUND, '
                                    '3: NO_DATA, 4: NO_DELTA_TIME , 5: OUTSIDE_ROI'}},
    'depth': {'layer_type': NOMINAL_LAYER,
              'layer_name': 'Depth',
              'dimensions': DIMS_Y_X_NKEEP_TIME,
              'data_type': np.float32,
              'fill_value': np.nan,
              'precision': 2,
              'attrs': {'Dimension': 'Meters [m]',
                        'name': 'Raw estimated depth'}},
    'direction_from_north': {'layer_type': NOMINAL_LAYER,
                             'layer_name': 'Direction',
                             'dimensions': DIMS_Y_X_NKEEP_TIME,
                             'data_type': np.float32,
                             'fill_value': np.nan,
                             'precision': 1,
                             'attrs': {'Dimension': 'degree',
                                       'name': 'Wave_direction'}},
    'celerity': {'layer_type': NOMINAL_LAYER,
                 'layer_name': 'Celerity',
                 'dimensions': DIMS_Y_X_NKEEP_TIME,
                 'data_type': np.float32,
                 'fill_value': np.nan,
                 'precision': 2,
                 'attrs': {'Dimension': 'Meters per second [m/sec]',
                           'name': 'Wave_celerity'}},
    'delta_position': {'layer_type': EXPERT_LAYER,
                       'layer_name': 'Propagated distance',
                       'dimensions': DIMS_Y_X_NKEEP_TIME,
                       'data_type': np.float32,
                       'fill_value': np.nan,
                       'precision': 2,
                       'attrs': {'Dimension': 'Meters [m]',
                                 'name': 'Distance used for measuring wave celerity'}},
    'absolute_delta_position': {'layer_type': EXPERT_LAYER,
                                'layer_name': 'Propagated distance (Absolute)',
                                'dimensions': DIMS_Y_X_NKEEP_TIME,
                                'data_type': np.float32,
                                'fill_value': np.nan,
                                'precision': 2,
                                'attrs': {'Dimension': 'Meters [m]',
                                          'name': 'Distance used for measuring wave celerity'}},
    'wavelength': {'layer_type': NOMINAL_LAYER,
                   'layer_name': 'Wavelength',
                   'dimensions': DIMS_Y_X_NKEEP_TIME,
                   'data_type': np.float32,
                   'fill_value': np.nan,
                   'precision': 1,
                   'attrs': {'Dimension': 'Meters [m]',
                             'name': 'Wavelength'}},
    'wavenumber': {'layer_type': EXPERT_LAYER,
                   'layer_name': 'Wavenumber',
                   'dimensions': DIMS_Y_X_NKEEP_TIME,
                   'data_type': np.float32,
                   'fill_value': np.nan,
                   'precision': 5,
                   'attrs': {'Dimension': 'Per Meter [m-1]',
                             'name': 'Wavenumber'}},
    'period': {'layer_type': EXPERT_LAYER,
               'layer_name': 'Period',
               'dimensions': DIMS_Y_X_NKEEP_TIME,
               'data_type': np.float32,
               'fill_value': np.nan,
               'precision': 2,
               'attrs': {'Dimension': 'Seconds [sec]',
                         'name': 'Wave_period'}},
    'distance_to_shore': {'layer_type': EXPERT_LAYER,
                          'layer_name': 'Distoshore',
                          'dimensions': DIMS_Y_X_TIME,
                          'data_type': np.float32,
                          'fill_value': np.nan,
                          'precision': 3,
                          'attrs': {'Dimension': 'Kilometers [km]',
                                    'name': 'Distance_to_shore'}},
    'delta_celerity': {'layer_type': EXPERT_LAYER,
                       'layer_name': 'Delta Celerity',
                       'dimensions': DIMS_Y_X_NKEEP_TIME,
                       'data_type': np.float32,
                       'fill_value': np.nan,
                       'precision': 2,
                       'attrs': {'Dimension': 'Meters per seconds2 [m/sec2]',
                                 'name': 'delta_celerity'}},
    'delta_phase': {'layer_type': EXPERT_LAYER,
                    'layer_name': 'Phase Shift',
                    'dimensions': DIMS_Y_X_NKEEP_TIME,
                    'data_type': np.float32,
                    'fill_value': np.nan,
                    'precision': 8,
                    'attrs': {'Dimension': 'Radians [rd]',
                              'name': 'Phase shift'}},
    'absolute_delta_phase': {'layer_type': EXPERT_LAYER,
                             'layer_name': 'Phase Shift (Absolute)',
                             'dimensions': DIMS_Y_X_NKEEP_TIME,
                             'data_type': np.float32,
                             'fill_value': np.nan,
                             'precision': 8,
                             'attrs': {'Dimension': 'Radians [rd]',
                                       'name': 'Phase shift (Absolute)'}},
    'gravity': {'layer_type': EXPERT_LAYER,
                'layer_name': 'Gravity',
                'dimensions': DIMS_Y_X_TIME,
                'data_type': np.float32,
                'fill_value': np.nan,
                'precision': 4,
                'attrs': {'Dimension': 'Acceleration [m/s2]',
                          'name': 'Gravity'}},
    'delta_time': {'layer_type': EXPERT_LAYER,
                   'layer_name': 'Delta Acquisition Time',
                   'dimensions': DIMS_Y_X_TIME,
                   'data_type': np.float32,
                   'fill_value': np.nan,
                   'precision': 4,
                   'attrs': {'Dimension': 'Duration (s)',
                             'name': 'DeltaTime'}},
    'ambiguity': {'layer_type': EXPERT_LAYER,
                  'layer_name': 'Ambiguity',
                  'dimensions': DIMS_Y_X_NKEEP_TIME,
                  'data_type': np.float32,
                  'fill_value': np.nan,
                  'precision': 4,
                  'attrs': {'Dimension': 'Unitless',
                            'name': 'delta_time / period'}},
    'absolute_ambiguity': {'layer_type': EXPERT_LAYER,
                           'layer_name': 'Ambiguity (absolute)',
                           'dimensions': DIMS_Y_X_NKEEP_TIME,
                           'data_type': np.float32,
                           'fill_value': np.nan,
                           'precision': 4,
                           'attrs': {'Dimension': 'Unitless',
                                     'name': '|delta_time| / period'}},
    'linearity': {'layer_type': EXPERT_LAYER,
                  'layer_name': 'Wave Linearity',
                  'dimensions': DIMS_Y_X_NKEEP_TIME,
                  'data_type': np.float32,
                  'fill_value': np.nan,
                  'precision': 3,
                  'attrs': {'Dimension': 'Unitless',
                            'name': 'linearity'}},
    'period_offshore': {'layer_type': EXPERT_LAYER,
                        'layer_name': 'Period Offshore',
                        'dimensions': DIMS_Y_X_NKEEP_TIME,
                        'data_type': np.float32,
                        'fill_value': np.nan,
                        'precision': 2,
                        'attrs': {'Dimension': 'Seconds [sec]',
                                  'name': 'Period of the wave field if it was offshore'}},
    'energy': {'layer_type': DEBUG_LAYER,
               'layer_name': 'Energy',
               'dimensions': DIMS_Y_X_NKEEP_TIME,
               'data_type': np.float32,
               'fill_value': np.nan,
               'precision': 1,
               'attrs': {'Dimension': 'Joules per Meter2 [J/m2]',
                         'name': 'Energy'}},
    'period_ratio': {'layer_type': DEBUG_LAYER,
                     'layer_name': 'Period Ratio',
                     'dimensions': DIMS_Y_X_NKEEP_TIME,
                     'data_type': np.float32,
                     'fill_value': np.nan,
                     'precision': 3,
                     'attrs': {'Dimension': 'Unitless',
                               'name': 'period_offshore / period'}},
    'energy_ratio': {'layer_type': DEBUG_LAYER,
                     'layer_name': 'Energy Ratio',
                     'dimensions': DIMS_Y_X_NKEEP_TIME,
                     'data_type': np.float32,
                     'fill_value': np.nan,
                     'precision': 3,
                     'attrs': {'Dimension': 'Joules per Meter2 [J/m2]',
                               'name': 'energy_ratio'}},
}


class EstimatedBathy:
    """ This class gathers all the estimated bathymetry samples in a whole dataset.
    """

    def __init__(self, x_samples: np.ndarray, y_samples: np.ndarray,
                 acq_time: str) -> None:
        """ Define dimensions for which the estimated bathymetry samples will be defined.

        :param x_samples: the X coordinates defining the estimated bathymetry samples
        :param y_samples: the Y coordinates defining the estimated bathymetry samples
        :param acq_time: the time at which the bathymetry samples are estimated
        """
        # data is stored as a 2D array of python objects, here a dictionary containing bathy fields.
        self.estimated_bathy = np.empty((y_samples.shape[0], x_samples.shape[0]), dtype=np.object_)

        timestamp = datetime(int(acq_time[:4]), int(acq_time[4:6]), int(acq_time[6:8]),
                             int(acq_time[9:11]), int(acq_time[11:13]), int(acq_time[13:15]))
        self.timestamps = [timestamp]
        self.x_samples = x_samples
        self.y_samples = y_samples

    def store_estimations(self, bathy_estimations: BathymetrySampleEstimations) -> None:
        """ Store a set of bathymetry estimations at some location

        :param bathy_estimations: the whole set of bathy estimations data at one point.
        :raises WavesEstimationIndexingError: when the x, y sample coordinates cannot be retrieved
        """
        x_sample, y_sample = bathy_estimations.location
        x_index = np.where(self.x_samples == x_sample)
        y_index = np.where(self.y_samples == y_sample)
        if len(x_index[0]) == 0 or len(y_index[0]) == 0:
            msg_err = f'x_sample: {x_sample} or y_sample: {y_sample} indexes not found'
            raise WavesEstimationIndexingError(msg_err)
        self.estimated_bathy[y_index[0][0], x_index[0][0]] = bathy_estimations

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
        for sample_property, layer_definition in BATHY_PRODUCT_DEF.items():
            if layers_type in layer_definition['layer_type']:
                try:
                    data_array = self._build_data_array(sample_property, layer_definition, nb_keep)
                    data_arrays[layer_definition['layer_name']] = data_array
                except WavesEstimationAttributeError:
                    # property was not found at any location: ignore it
                    continue

        # Combine all DataArray in a single Dataset:
        return Dataset(data_vars=data_arrays)

    def _build_data_array(self, sample_property: str,
                          layer_definition: Dict[str, Any], nb_keep: int) -> DataArray:
        """ Build an xarray DataArray containing one estimated bathymetry property.

        :param sample_property: name of the property to format as a DataArray
        :param layer_definition: definition of the way to format the property
        :param nb_keep: the number of different bathymetry estimations to keep for one location.
        :raises IndexError: when the property is not a scalar or a vector
        :returns: an xarray DataArray containing the formatted property
        """
        nb_samples_y, nb_samples_x = self.estimated_bathy.shape

        dims = layer_definition['dimensions']
        if 'kKeep' in dims:
            layer_shape = (nb_samples_y, nb_samples_x, nb_keep)
        else:
            layer_shape = (nb_samples_y, nb_samples_x)
        layer_data = np.full(layer_shape,
                             layer_definition['fill_value'],
                             dtype=layer_definition['data_type'])

        not_found = 0
        for y_index in range(nb_samples_y):
            for x_index in range(nb_samples_x):
                try:
                    self._fill_array(sample_property, layer_data, y_index, x_index)
                except WavesEstimationAttributeError:
                    not_found += 1
                    continue
        if not_found == nb_samples_x * nb_samples_y:
            raise WavesEstimationAttributeError(f'no values defined for: {sample_property}')

        layer_data.round(layer_definition['precision'])
        # Add a dimension at the end for time singleton
        array = np.expand_dims(layer_data, axis=layer_data.ndim)
        return DataArray(array, coords=self._get_coords(dims, nb_keep),
                         dims=dims, attrs=layer_definition['attrs'])

    # TODO: split array filling in two methods: one for 2D (X, Y) and one for 3D (X, Y, kKeep)
    def _fill_array(self, sample_property: str, layer_data: np.ndarray,
                    y_index: int, x_index: int) -> None:
        wave_fields_estimations = self.estimated_bathy[y_index, x_index]
        bathy_property = wave_fields_estimations.get_attribute(sample_property)

        if layer_data.ndim == 2:
            layer_data[y_index, x_index] = np.array(bathy_property)
        else:
            nb_keep = layer_data.shape[2]
            if len(bathy_property) > nb_keep:
                bathy_property = bathy_property[:nb_keep]
            elif len(bathy_property) < nb_keep:
                bathy_property += [np.nan] * (nb_keep - len(bathy_property))
            layer_data[y_index, x_index, :] = np.array(bathy_property)

    def _get_coords(self, dims: List[str], nb_keep: int) -> Mapping[Hashable, Any]:
        dict_coords: Mapping[Hashable, Any] = {}
        for element in dims:
            if element == 'y':
                value = self.y_samples
            elif element == 'x':
                value = self.x_samples
            elif element == 'kKeep':
                value = np.arange(1, nb_keep + 1)
            else:
                value = self.timestamps
            dict_coords[element] = value
        return dict_coords
