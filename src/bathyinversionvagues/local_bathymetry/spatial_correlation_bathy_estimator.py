# -*- coding: utf-8 -*-
"""
Class performing bathymetry computation using spatial correlation method

@author: erwinbergsma
         gregoirethoumyre
         degoulromain
"""

from typing import Optional, List, TYPE_CHECKING

import numpy as np
from munch import Munch

from ..image_processing.waves_image import WavesImage, ImageProcessingFilters
from ..image_processing.waves_radon import SignalProcessingFilters
from ..local_bathymetry.correlation_bathy_estimator import CorrelationBathyEstimator
from ..generic_utils.image_utils import normxcorr2
from ..generic_utils.image_filters import detrend, clipping
from ..generic_utils.signal_filters import filter_mean, remove_median

if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport

class SpatialCorrelationBathyEstimator(CorrelationBathyEstimator):
    def __init__(self, images_sequence: List[WavesImage], global_estimator: 'BathyEstimator',
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
        sampling_positions_x = np.reshape(np.arange(self._shape_x), (1, -1))
        sampling_positions_y = np.reshape(np.arange(self._shape_y), (1, -1))
        self._sampling_positions = (sampling_positions_x,sampling_positions_y)

    @property
    def _parameters(self) -> Munch:
        """
        :return: munchified parameters
        """
        return self.local_estimator_params.TEMPORAL_METHOD

    @property
    def sampling_positions(self) -> np.ndarray:
        """
        :return: tuple of sampling positions
        """
        return self._sampling_positions

    def get_correlation_matrix(self) -> np.ndarray:
        """
        :return: correlation matrix
        """
        merge_array = np.dstack([image.pixels for image in self.images_sequence])
        full_corr = normxcorr2(merge_array[:, :, 0].T,
                               merge_array[:, :, self._parameters.TEMPORAL_LAG].T)
        for index in np.arange(self._parameters.TEMPORAL_LAG, self._number_frames -
                                                              self._parameters.TEMPORAL_LAG,
                               self._parameters.TEMPORAL_LAG):
            corr = normxcorr2(merge_array[:, :, index].T,
                              merge_array[:, :, index + self._parameters.TEMPORAL_LAG].T)
            full_corr = full_corr + corr
        return full_corr
