# -*- coding: utf-8 -*-
""" Base class for the estimators of waves fields from several images taken at a small
time intervals.

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 5 mars 2021
"""
from abc import abstractmethod, ABC
from copy import deepcopy

from typing import Dict, Any, List, Optional  # @NoMove

import numpy as np

from bathyinversionvagues.image.image_geometry_types import PointType

from ..image_processing.waves_image import WavesImage
from .waves_field_estimation import WavesFieldEstimation

WavesFieldsEstimations = List[WavesFieldEstimation]


class LocalBathyEstimator(ABC):
    def __init__(self, images_sequence: List[WavesImage], global_estimator,
                 selected_directions: Optional[np.ndarray] = None) -> None:
        """ Constructor

        :param selected_directions: the set of directions onto which the sinogram must be computed
        """
        # TODO: Check that the images have the same resolution, satellite (and same size ?)
        self.global_estimator = global_estimator
        self.local_estimator_params = self.global_estimator.waveparams

        self.images_sequence = images_sequence
        self.selected_directions = selected_directions

        self._position = (0., 0.)
        self._gravity = 0.
        self._waves_fields_estimations: WavesFieldsEstimations = []

        self._metrics: Dict[str, Any] = {}

    def set_position(self, point: PointType) -> None:
        self._position = point
        self._gravity = self.global_estimator.get_gravity(self._position, 0.)

    @property
    def gravity(self) -> float:
        return self._gravity

    @abstractmethod
    def run(self) -> None:
        """  Run the local bathymetry estimation, using some method specific to the inheriting
        class.

        This method stores its results using the store_estimation() method and
        its metrics in _metrics attribute.
        """

    def create_waves_field_estimation(self, direction: float, wavelength: float
                                      ) -> WavesFieldEstimation:
        # FIXME: DT is not the right value to take into account. Use
        # DeltaTimeProvider when written. The same for gravity
        waves_field_estimation = WavesFieldEstimation(self.local_estimator_params.DT,
                                                      self.local_estimator_params.D_PRECISION,
                                                      self.gravity,
                                                      self.local_estimator_params.DEPTH_EST_METHOD)
        waves_field_estimation.direction = direction
        waves_field_estimation.wavelength = wavelength

        return waves_field_estimation

    def store_estimation(self, waves_field_estimation: WavesFieldEstimation) -> None:
        """ Store a single estimation into the estimations list

        :param waves_field_estimation: a new estimation to store for this local bathy estimator
        """
        self._waves_fields_estimations.append(waves_field_estimation)

    @property
    def waves_fields_estimations(self) -> WavesFieldsEstimations:
        """ :returns: a copy of the estimations recorded by this estimator.
                      Used for freeing references to memory expensive data (images, transform, ...)
        """
        return deepcopy(self._waves_fields_estimations)

    @property
    def metrics(self) -> Dict[str, Any]:
        """ :returns: a copy of the dictionary of metrics recorded by this estimator.
                      Used for freeing references to memory expensive data (images, transform, ...)
        """
        return deepcopy(self._metrics)

    def print_estimations_debug(self, step: str) -> None:
        self.global_estimator.print_estimations_debug(self._waves_fields_estimations, step)
