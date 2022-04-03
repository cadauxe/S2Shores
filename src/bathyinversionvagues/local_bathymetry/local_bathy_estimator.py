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

from typing import Dict, Any, List, Optional, Type, TYPE_CHECKING  # @NoMove

import numpy as np

from ..data_model.bathymetry_sample_estimation import BathymetrySampleEstimation
from ..data_model.bathymetry_sample_estimations import BathymetrySampleEstimations
from ..image.image_geometry_types import PointType
from ..image_processing.waves_image import WavesImage, ImageProcessingFilters
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
    def waves_field_estimation_cls(cls) -> Type[BathymetrySampleEstimation]:
        """ :returns: a class inheriting from BathymetrySampleEstimation to use for storing an
                      estimation.
        """

    def __init__(self, location: PointType, global_estimator: 'BathyEstimator',
                 selected_directions: Optional[np.ndarray] = None) -> None:
        """ Constructor

        :param location: The (X, Y) coordinates of the location where this estimator is acting
        :param global_estimator: a global bathymetry estimator able to provide the services needed
                                 by this local bathymetry estimator (access to parameters,
                                 data providers, debugging, ...)
        :param selected_directions: the set of directions onto which the sinogram must be computed
        :raise SequenceImagesError: when sequence can no be exploited
        """
        self.images_sequence: List[WavesImage] = []
        self.spatial_resolution = 0.

        self.global_estimator = global_estimator
        self.debug_sample = self.global_estimator.debug_sample
        self.local_estimator_params = self.global_estimator.local_estimator_params

        self.selected_directions = selected_directions

        # FIXME: distance to shore test should take into account windows sizes
        distance = self.global_estimator.get_distoshore(location)
        gravity = self.global_estimator.get_gravity(location, 0.)
        inside_roi = self.global_estimator.is_inside_roi(location)
        self._waves_fields_estimations = BathymetrySampleEstimations(location, gravity,
                                                                     distance, inside_roi)

        sequential_delta_times = np.array([])
        for frame_index in range(len(self.global_estimator.selected_frames) - 1):
            delta_time = self.global_estimator.get_delta_time(
                self.global_estimator.selected_frames[frame_index],
                self.global_estimator.selected_frames[frame_index + 1],
                self.location)
            # FIXME: copied from CorrelationBathyEstimator but wrong !?
            sequential_delta_times = np.append(delta_time, sequential_delta_times)
        self._sequential_delta_times = sequential_delta_times

        self._metrics: Dict[str, Any] = {}

    def can_estimate_bathy(self) -> bool:
        return (self.waves_fields_estimations.distance_to_shore > 0 and
                self.waves_fields_estimations.inside_roi)

    def set_images_sequence(self, images_sequence: List[WavesImage]) -> None:
        """ initialize the image_sequence to use with this estimator

        :param images_sequence: a list of superimposed local images centered around the position
                                where the estimator is working.
        :raise SequenceImagesError: when sequence can no be exploited
        """
        if self.spatial_resolution != 0.:
            raise SequenceImagesError('Cannot redefine the sequence of images for this estimator')
        if not images_sequence:
            raise SequenceImagesError('Sequence images is empty')
        self.spatial_resolution = images_sequence[0].resolution
        shape = images_sequence[0].pixels.shape
        for wave_image in images_sequence[1:]:
            if wave_image.resolution != self.spatial_resolution:
                raise SequenceImagesError('Images in sequence do not have same resolution')
            if wave_image.pixels.shape != shape:
                raise SequenceImagesError('Images in sequence do not have same size')
        self.images_sequence = images_sequence

    @property
    @abstractmethod
    def propagation_duration(self) -> float:
        """ :returns: The time length of the sequence of images used for the estimation. May be
                      positive or negative to account for chronology of start and stop images.
        """

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
        for image in self.images_sequence:
            filtered_image = image.apply_filters(self.preprocessing_filters)
            image.pixels = filtered_image.pixels

    @property
    def gravity(self) -> float:
        """ :returns: the acceleration of the gravity at the working position of the estimator
        """
        return self.waves_fields_estimations.gravity

    @property
    def location(self) -> PointType:
        """ :returns: The (X, Y) coordinates of the location where this estimator is acting"""
        return self.waves_fields_estimations.location

    @property
    def sequential_delta_times(self) -> np.ndarray:
        """ :returns: the time differences between 2 consecutive frames in the image sequence
        """
        return self._sequential_delta_times

    @abstractmethod
    def run(self) -> None:
        """  Run the local bathymetry estimation, using some method specific to the inheriting
        class.

        This method stores its results in the waves_fields_estimations list and
        its metrics in _metrics attribute.
        """

    def create_waves_field_estimation(self, direction: float, wavelength: float
                                      ) -> BathymetrySampleEstimation:
        """ Creates the BathymetrySampleEstimation instance where the local estimator will store its
        estimation.

        :param direction: the propagation direction of the waves field (degrees measured
                          counterclockwise from the East).
        :param wavelength: the wavelength of the waves field
        :returns: an initialized instance of WavesFilesEstimation to be filled in further on.
        """
        waves_field_estimation = self.waves_field_estimation_cls(
            self.gravity,
            self.global_estimator.depth_estimation_method,
            self.global_estimator.waves_period_range,
            self.global_estimator.waves_linearity_range,
            self.global_estimator.depth_min)
        waves_field_estimation.delta_time = self.propagation_duration
        waves_field_estimation.direction = direction
        waves_field_estimation.wavelength = wavelength

        return waves_field_estimation

    @property
    def waves_fields_estimations(self) -> BathymetrySampleEstimations:
        """ :returns: the estimations recorded by this estimator.
        """
        return self._waves_fields_estimations

    @property
    def metrics(self) -> Dict[str, Any]:
        """ :returns: a copy of the dictionary of metrics recorded by this estimator.
                      Used for freeing references to memory expensive data (images, transform, ...)
        """
        return self._metrics

    @metrics.setter
    def metrics(self, values: Dict[str, Any]) -> None:
        self._metrics = deepcopy(values)
