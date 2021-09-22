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

from typing import Dict, Any, List, Optional, TYPE_CHECKING  # @NoMove

import numpy as np

from ..image_processing.waves_image import WavesImage, ImageProcessingFilters
from ..sequence_images_exceptions import (SequenceImagesSizeError, SequenceImagesResolutionError,
                                          SequenceImagesEmptyError)
from .waves_field_estimation import WavesFieldEstimation
from .waves_fields_estimations import WavesFieldsEstimations


if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport


class LocalBathyEstimator(ABC):
    """ Abstract base class of all local bathymetry estimators.
    """

    def __init__(self, images_sequence: List[WavesImage], global_estimator: 'BathyEstimator',
                 waves_fields_estimations: WavesFieldsEstimations,
                 selected_directions: Optional[np.ndarray] = None) -> None:
        """ Constructor

        :param images_sequence: a list of superimposed local images centered around the position
                                where the estimator is working.
        :param global_estimator: a global bathymetry estimator able to provide the services needed
                                 by this local bathymetry estimator (access to parameters,
                                 data providers, debugging, ...)
        :param selected_directions: the set of directions onto which the sinogram must be computed
        :raise SequenceImagesEmptyError: when sequence is empty
        :raise SequenceImagesResolutionError: when sequence has different resolutions
        :raise SequenceImagesSizeError: when sequence has different sizes
        """
        if not images_sequence:
            raise SequenceImagesEmptyError('Sequence images is empty')
        spatial_resolution = images_sequence[0].resolution
        shape = images_sequence[0].pixels.shape
        for wave_image in images_sequence[1:]:
            if wave_image.resolution != spatial_resolution:
                raise SequenceImagesResolutionError(
                    'Images in sequence do not have same resolution')
            if wave_image.pixels.shape != shape:
                raise SequenceImagesSizeError('Images in sequence do not have same size')

        self.spatial_resolution = spatial_resolution

        self.global_estimator = global_estimator
        self.debug_sample = self.global_estimator.debug_sample
        self.local_estimator_params = self.global_estimator.local_estimator_params

        self.images_sequence = images_sequence
        self.selected_directions = selected_directions

        self._waves_fields_estimations = waves_fields_estimations
        self._position = self._waves_fields_estimations.location

        self._delta_time = self.global_estimator.get_delta_time(
            self._waves_fields_estimations.location)

        self._metrics: Dict[str, Any] = {}

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
            image.apply_filters(self.preprocessing_filters)

    @property
    def gravity(self) -> float:
        """ :returns: the acceleration of the gravity at the working position of the estimator
        """
        return self._waves_fields_estimations.gravity

    # FIXME: At the moment only a pair of images is handled (list is limited to a singleton)
    @property
    def delta_time(self) -> float:
        """ :returns: the time differences between 2 consecutive frames in the image sequence
        """
        return self._delta_time

    @abstractmethod
    def run(self) -> None:
        """  Run the local bathymetry estimation, using some method specific to the inheriting
        class.

        This method stores its results using the store_estimation() method and
        its metrics in _metrics attribute.
        """

    @abstractmethod
    def sort_waves_fields(self) -> None:
        """  Sorts the waves fields on whatever criteria.
        """

    def validate_waves_fields(self) -> None:
        """  Remove non physical waves fields
        """
        # Filter non physical waves fields and bathy estimations
        # We iterate over a copy of the list in order to keep waves_fields_etimations unaffected
        # on its specific attributes
        # for index, estimation in enumerate(list(self.waves_fields_estimations)):
        for estimation in list(self.waves_fields_estimations):
            if (estimation.period < self.global_estimator.waves_period_min or
                    estimation.period > self.global_estimator.waves_period_max):
                self.waves_fields_estimations.remove(estimation)
        for estimation in list(self.waves_fields_estimations):
            if (estimation.linearity < self.global_estimator.waves_linearity_min or
                    estimation.linearity > self.global_estimator.waves_linearity_max):
                self.waves_fields_estimations.remove(estimation)

    @abstractmethod
    def create_waves_field_estimation(self, direction: float, wavelength: float
                                      ) -> WavesFieldEstimation:
        """ Creates the WavesFieldEstimation instance where the local estimator will store its
        estimations.

        :param direction: the propagation direction of the waves field (degrees measured clockwise
                          from the North).
        :param wavelength: the wavelength of the waves field
        :returns: an initialized instance of WavesFilesEstimation to be filled in further on.
        """

    def store_estimation(self, waves_field_estimation: WavesFieldEstimation) -> None:
        """ Store a single estimation into the estimations list

        :param waves_field_estimation: a new estimation to store for this local bathy estimator
        """
        self._waves_fields_estimations.append(waves_field_estimation)

    @property
    def waves_fields_estimations(self) -> WavesFieldsEstimations:
        """ :returns: the estimations recorded by this estimator.
        """
        return self._waves_fields_estimations

    @property
    def metrics(self) -> Dict[str, Any]:
        """ :returns: a copy of the dictionary of metrics recorded by this estimator.
                      Used for freeing references to memory expensive data (images, transform, ...)
        """
        return deepcopy(self._metrics)

    def print_estimations_debug(self, step: str) -> None:
        """ Print debugging info on the estimations if the point has been tagged for debugging

        :param step: A string to be printed as header of the debugging info.
        """
        if self.debug_sample:
            print(f'estimations at step: {step}')
            print(self.waves_fields_estimations)


class LocalBathyEstimatorDebug(LocalBathyEstimator):
    """ Abstract class handling begud mode for LocalBathyEstimator
    """

    def run(self) -> None:
        super().run()
        if self.debug_sample:
            self.explore_results()

    @abstractmethod
    def explore_results(self) -> None:
        """ Method called when estimator has run to allow results exploration for debugging purposes
        """
