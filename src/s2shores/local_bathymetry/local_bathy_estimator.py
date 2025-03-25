# -*- coding: utf-8 -*-
""" Base class for the estimators of wave fields from several images taken at a small
time intervals.

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 5 mars 2021

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Optional, Type  # @NoMove

import numpy as np
from shapely.geometry import Point

from ..data_model.bathymetry_sample_estimation import BathymetrySampleEstimation
from ..data_model.bathymetry_sample_estimations import BathymetrySampleEstimations
from ..data_providers.delta_time_provider import NoDeltaTimeValueError
from ..image.ortho_sequence import FrameIdType, OrthoSequence
from ..image_processing.waves_image import ImageProcessingFilters
from ..waves_exceptions import SequenceImagesError

if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport


class LocalBathyEstimator(ABC):
    """ Abstract base class of all local bathymetry estimators.
    """

    final_estimations_sorting: Optional[str] = None

    @property
    @classmethod
    @abstractmethod
    def wave_field_estimation_cls(cls) -> Type[BathymetrySampleEstimation]:
        """ :returns: a class inheriting from BathymetrySampleEstimation to use for storing an
                      estimation.
        """

    def __init__(self, location: Point, ortho_sequence: OrthoSequence,
                 global_estimator: 'BathyEstimator',
                 selected_directions: Optional[np.ndarray] = None) -> None:
        """ Constructor

        :param location: The (X, Y) coordinates of the location where this estimator is acting
        :param ortho_sequence: a list of superimposed local images centered around the position
                                where the estimator is working.
        :param global_estimator: a global bathymetry estimator able to provide the services needed
                                 by this local bathymetry estimator (access to parameters,
                                 data providers, debugging, ...)
        :param selected_directions: the set of directions onto which the sinogram must be computed
        :raises SequenceImagesError: when the sequence is empty
        """
        if not ortho_sequence:
            raise SequenceImagesError('Ortho Sequence is empty')
        self.ortho_sequence = ortho_sequence
        self.spatial_resolution = ortho_sequence.resolution

        self._location = location
        self.global_estimator = global_estimator
        self.debug_sample = self.global_estimator.debug_sample
        self.local_estimator_params = self.global_estimator.local_estimator_params

        self.selected_directions = selected_directions

        # FIXME: distance to shore test should take into account windows sizes
        distance = self.global_estimator.get_distoshore(self.location)
        gravity = self.global_estimator.get_gravity(self.location, 0.)
        inside_roi = self.global_estimator.is_inside_roi(self.location)
        inside_offshore_limit = (distance <= self.global_estimator.max_offshore_distance or
                                 distance == np.inf)

        self._bathymetry_estimations = BathymetrySampleEstimations(self._location, gravity, np.nan,
                                                                   distance, inside_roi,
                                                                   inside_offshore_limit)
        try:
            propagation_duration = self.ortho_sequence.get_time_difference(self._location,
                                                                           self.start_frame_id,
                                                                           self.stop_frame_id)
            self.bathymetry_estimations.delta_time = propagation_duration
        except NoDeltaTimeValueError:
            self.bathymetry_estimations.delta_time = np.nan

        self._metrics: Dict[str, Any] = {}

    def can_estimate_bathy(self) -> bool:
        """ Test if conditions to start bathymetry estimation are met.

        :returns: True if the point is on water and inside a possible ROI and if delta time is
                  available for that point, False otherwise.
        """
        return (self.bathymetry_estimations.distance_to_shore > 0 and
                self.bathymetry_estimations.inside_offshore_limit and
                self.bathymetry_estimations.inside_roi and
                self.bathymetry_estimations.delta_time_available)

    @property
    @abstractmethod
    def start_frame_id(self) -> FrameIdType:
        """ :returns: The id of the start frame used by this estimator.
        """

    @property
    @abstractmethod
    def stop_frame_id(self) -> FrameIdType:
        """ :returns: The id of the stop frame used by this estimator.
        """

    @property
    def propagation_duration(self) -> float:
        """ :returns: The time length of the sequence of images used for the estimation. May be
                      positive or negative to account for chronology of start and stop images.
        """
        return self.bathymetry_estimations.delta_time

    @property
    @abstractmethod
    def preprocessing_filters(self) -> ImageProcessingFilters:
        """ :returns: A list of functions together with their parameters to be applied
        sequentially to all the images of the sequence before subsequent bathymetry estimation.
        """

    def preprocess_images(self) -> None:
        """ Process the images before doing the bathymetry estimation with a sequence of
        image processing filters.
        """
        for image in self.ortho_sequence:
            filtered_image = image.apply_filters(self.preprocessing_filters)
            image.pixels = filtered_image.pixels

    @property
    def gravity(self) -> float:
        """ :returns: the acceleration of the gravity at the working position of the estimator
        """
        return self.bathymetry_estimations.gravity

    @property
    def location(self) -> Point:
        """ :returns: The location where this estimator is acting"""
        return self._location

    @abstractmethod
    def run(self) -> None:
        """  Run the local bathymetry estimation, using some method specific to the inheriting
        class.

        This method stores its results in the bathymetry_estimations list and
        its metrics in _metrics attribute.
        """

    def create_bathymetry_estimation(self, direction: float, wavelength: float
                                     ) -> BathymetrySampleEstimation:
        """ Creates the BathymetrySampleEstimation instance where the local estimator will store its
        estimation.

        :param direction: the propagation direction of the wave field (degrees measured
                          counterclockwise from the East).
        :param wavelength: the wavelength of the wave field
        :returns: an initialized instance of BathymetrySampleEstimation to be filled in further on.
        """
        bathy_estimation = self.wave_field_estimation_cls(
            self.gravity,
            self.global_estimator.depth_estimation_method,
            self.global_estimator.waves_period_range,
            self.global_estimator.waves_linearity_range,
            self.global_estimator.depth_min)
        bathy_estimation.delta_time = self.propagation_duration
        bathy_estimation.direction = direction
        bathy_estimation.wavelength = wavelength

        return bathy_estimation

    @property
    def bathymetry_estimations(self) -> BathymetrySampleEstimations:
        """ :returns: the wave fields estimations recorded by this estimator.
        """
        return self._bathymetry_estimations

    @property
    def metrics(self) -> Dict[str, Any]:
        """ :returns: a copy of the dictionary of metrics recorded by this estimator.
                      Used for freeing references to memory expensive data (images, transform, ...)
        """
        return self._metrics

    @metrics.setter
    def metrics(self, values: Dict[str, Any]) -> None:
        self._metrics = deepcopy(values)
