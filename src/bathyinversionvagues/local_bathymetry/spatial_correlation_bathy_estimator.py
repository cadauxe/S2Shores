# -*- coding: utf-8 -*-
"""
Class performing bathymetry computation using spatial correlation method

@author: erwinbergsma
         gregoirethoumyre
         degoulromain
"""

from typing import Optional, List
from munch import Munch

import numpy as np

from ..image_processing.waves_image import WavesImage
from ..local_bathymetry.correlation_bathy_estimator import CorrelationBathyEstimator
from ..image_processing.shoresutils import normxcorr2


class SpatialCorrelationBathyEstimator(CorrelationBathyEstimator):
    def __init__(self, images_sequence: List[WavesImage], global_estimator,
                 selected_directions: Optional[np.ndarray] = None) -> None:
        """
        :param images_sequence: sequence of image used to compute bathymetry
        :param global_estimator: global estimator
        :param selected_directions: selected_directions: the set of directions onto which the
        sinogram must be computed
        """
        super().__init__(images_sequence, global_estimator, selected_directions)
        self._shape_x, self._shape_y = self.images_sequence[0].pixels.shape
        self._number_frames = len(self.images_sequence)
        self._positions_x = np.reshape(np.arange(self._shape_x), (1, -1))
        self._positions_y = np.reshape(np.arange(self._shape_y), (1, -1))

    @property
    def _parameters(self) -> Munch:
        """
        :return: munchified parameters
        """
        return self.local_estimator_params.SPATIAL_METHOD

    def get_correlation_matrix(self) -> np.ndarray:
        """
        :return: correlation matrix
        """
        merge_array = np.dstack([image.pixels for image in self.images_sequence])
        full_corr = normxcorr2(merge_array[:, :, 0].T,
                               merge_array[:, :, self._parameters.TEMPORAL_LAG].T)
        for index in np.arange(self._parameters.TEMPORAL_LAG, self._number_frames -
                self._parameters.TEMPORAL_LAG, self._parameters.TEMPORAL_LAG):
            corr = normxcorr2(merge_array[:, :, index].T,
                              merge_array[:, :, index + self._parameters.TEMPORAL_LAG].T)
            full_corr = full_corr + corr
        return full_corr