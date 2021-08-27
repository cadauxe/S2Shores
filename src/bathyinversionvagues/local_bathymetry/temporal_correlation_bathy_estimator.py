# -*- coding: utf-8 -*-
"""
Class performing bathymetry computation using temporal correlation method

@author: erwinbergsma
         gregoirethoumyre
         degoulromain
"""

from typing import Optional, List
from munch import Munch

import numpy as np
import pandas

from ..image_processing.waves_image import WavesImage
from ..local_bathymetry.correlation_bathy_estimator import CorrelationBathyEstimator
from ..image_processing.shoresutils import cross_correlation


class TemporalCorrelationBathyEstimator(CorrelationBathyEstimator):
    def __init__(self, images_sequence: List[WavesImage], global_estimator,
                 selected_directions: Optional[np.ndarray] = None) -> None:
        """
        :param images_sequence: sequence of image used to compute bathymetry
        :param global_estimator: global estimator
        :param selected_directions: selected_directions: the set of directions onto which the
        sinogram must be computed
        """
        super().__init__(images_sequence, global_estimator, selected_directions)
        self._time_series = None
        self.create_sequence_time_series()

    def create_sequence_time_series(self):
        """
        This function computes an np.array of time series.
        To do this random points are selected within the sequence of image and a temporal serie
        is included in the np.array for each selected point
        """
        if self._parameters.PERCENTAGE_POINTS < 0 or self._parameters.PERCENTAGE_POINTS > 100:
            raise ValueError("Percentage must be between 0 and 100")
        merge_array = np.dstack([image.pixels for image in self.images_sequence])
        shape_x, shape_y = self.images_sequence[0].pixels.shape
        time_series = np.reshape(merge_array, (shape_x * shape_y, -1))
        nb_random_points = round(shape_x * shape_y * self._parameters.PERCENTAGE_POINTS / 100)
        random_indexes = np.random.randint(0, shape_x * shape_y, size=nb_random_points)
        positions_y, positions_x = np.meshgrid(np.linspace(1, shape_x, shape_x),
                                               np.linspace(1, shape_y, shape_y))
        self._positions_x = np.reshape(positions_x.flatten()[random_indexes], (1, -1))
        self._positions_y = np.reshape(positions_y.flatten()[random_indexes], (1, -1))
        self._time_series = time_series[random_indexes, :]

    def get_correlation_matrix(self) -> np.ndarray:
        """
        Compute temporal correlation matrix
        """
        return cross_correlation(self._time_series[:, self._parameters.TEMPORAL_LAG:],
                                 self._time_series[:, :-self._parameters.TEMPORAL_LAG])

    def get_correlation_image(self) -> WavesImage:
        """
        This function computes the correlation image by projecting the the correlation matrix on an
        array where axis are distances and center is the point where distance is 0.
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
        return WavesImage(projected_matrix, self._parameters.RESOLUTION.SPATIAL)

    @property
    def _parameters(self) -> Munch:
        """
        :return: munchified parameters
        """
        return self.local_estimator_params.TEMPORAL_METHOD
