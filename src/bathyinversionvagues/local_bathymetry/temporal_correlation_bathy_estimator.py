# -*- coding: utf-8 -*-
""" Class performing bathymetry computation using temporal correlation method

:author: Degoul Romain
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 18/06/2021
"""
from typing import Optional, List, Tuple, TYPE_CHECKING  # @NoMove

import pandas

import numpy as np

from ..bathy_debug.debug_display import temporal_method_debug
from ..generic_utils.image_utils import cross_correlation
from ..image_processing.waves_image import WavesImage
from ..local_bathymetry.correlation_bathy_estimator import CorrelationBathyEstimator
from ..local_bathymetry.local_bathy_estimator import LocalBathyEstimatorDebug
from .waves_fields_estimations import WavesFieldsEstimations


if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport


class TemporalCorrelationBathyEstimator(CorrelationBathyEstimator):
    """ Class performing temporal correlation to compute bathymetry
    """

    def __init__(self, images_sequence: List[WavesImage], global_estimator: 'BathyEstimator',
                 waves_fields_estimations: WavesFieldsEstimations,
                 selected_directions: Optional[np.ndarray] = None) -> None:

        super().__init__(images_sequence, global_estimator,
                         waves_fields_estimations, selected_directions)
        self.create_sequence_time_series()

    def create_sequence_time_series(self) -> None:
        """ This function computes an np.array of time series.
        To do this random points are selected within the sequence of image and a temporal serie
        is included in the np.array for each selected point
        """
        percentage_points = self.local_estimator_params['PERCENTAGE_POINTS']
        if percentage_points < 0 or percentage_points > 100:
            raise ValueError('Percentage must be between 0 and 100')
        merge_array = np.dstack([image.pixels for image in self.images_sequence])
        shape_x, shape_y = self.images_sequence[0].pixels.shape
        time_series = np.reshape(merge_array, (shape_x * shape_y, -1))
        nb_random_points = round(shape_x * shape_y * percentage_points / 100)
        random_indexes = np.random.randint(0, shape_x * shape_y, size=nb_random_points)
        positions_y, positions_x = np.meshgrid(np.linspace(1, shape_x, shape_x),
                                               np.linspace(1, shape_y, shape_y))

        sampling_positions_x = np.reshape(positions_x.flatten()[random_indexes], (1, -1))
        sampling_positions_y = np.reshape(positions_y.flatten()[random_indexes], (1, -1))
        self._sampling_positions = (sampling_positions_x, sampling_positions_y)
        self._time_series = time_series[random_indexes, :]

    def get_correlation_matrix(self) -> np.ndarray:
        """Compute temporal correlation matrix
        """
        temporal_lag = self.local_estimator_params['TEMPORAL_LAG']
        return cross_correlation(self._time_series[:, temporal_lag:],
                                 self._time_series[:, :-temporal_lag])

    def get_correlation_image(self) -> WavesImage:
        """ This function computes the correlation image by projecting the the correlation matrix
        on an array where axis are distances and center is the point where distance is 0.
        If several points have same coordinates, the mean of correlation is taken for this position
        """

        indices_x = np.round(self.distances * np.cos(np.radians(self.angles)))
        indices_x = np.array(indices_x - np.min(indices_x), dtype=int).T

        indices_y = np.round(self.distances * np.sin(np.radians(self.angles)))
        indices_y = np.array(indices_y - np.min(indices_y), dtype=int).T

        xr_s = pandas.Series(indices_x.flatten())
        yr_s = pandas.Series(indices_y.flatten())
        values_s = pandas.Series(self.correlation_matrix.flatten())

        # if two correlation values have same xr and yr mean of these values is taken
        dataframe = pandas.DataFrame({'xr': xr_s, 'yr': yr_s, 'values': values_s})
        dataframe_grouped = dataframe.groupby(by=['xr', 'yr']).mean().reset_index()
        values = np.array(dataframe_grouped['values'])
        indices_x = np.array(dataframe_grouped['xr'])
        indices_y = np.array(dataframe_grouped['yr'])

        projected_matrix = np.nanmean(self.correlation_matrix) * np.ones(
            (np.max(indices_x) + 1, np.max(indices_y) + 1))
        projected_matrix[indices_x, indices_y] = values
        return WavesImage(projected_matrix, self.spatial_resolution)

    @property
    def sampling_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """ :return: tuple of sampling positions
        """
        return self._sampling_positions


class TemporalCorrelationBathyEstimatorDebug(LocalBathyEstimatorDebug,
                                             TemporalCorrelationBathyEstimator):
    """ Class performing debugging for temporal correlation method
    """

    def explore_results(self) -> None:
        temporal_method_debug(self)
