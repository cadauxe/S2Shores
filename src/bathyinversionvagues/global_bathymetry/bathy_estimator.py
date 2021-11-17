# -*- coding: utf-8 -*-
""" Definition of the BathyEstimator abstract class

:author: GIROS Alain
:created: 17/05/2021
"""
from abc import ABC

from typing import List, Optional, Tuple  # @NoMove

from xarray import Dataset  # @NoMove
from munch import Munch
import numpy as np

from ..data_providers.delta_time_provider import (
    DeltaTimeProvider, NoDeltaTimeProviderError)
from ..data_providers.dis_to_shore_provider import InfinityDisToShoreProvider, DisToShoreProvider
from ..data_providers.gravity_provider import LatitudeVaryingGravityProvider, GravityProvider
from ..image.image_geometry_types import MarginsType, PointType
from ..image.ortho_stack import OrthoStack, FrameIdType, FramesIdsType
from ..image.sampled_ortho_image import SampledOrthoImage

from .bathy_estimator_parameters import BathyEstimatorParameters
from .ortho_bathy_estimator import OrthoBathyEstimator


class BathyEstimator(ABC, BathyEstimatorParameters):
    """ Management of bathymetry computation and parameters on a single product. Computation
    is split in several cartographic tiles, which must be run separately, either in parallel or
    sequentially.
    """

    def __init__(self, ortho_stack: OrthoStack, wave_params: Munch,
                 nb_subtiles_max: int = 1) -> None:
        """Create a BathyEstimator object and set necessary informations

        :param ortho_stack: the orthorectified stack onto which bathymetry must be estimated.
        :param wave_params: parameters for the global and local bathymetry estimators
        :param nb_subtiles_max: Nb of subtiles for bathymetry estimation
        """
        super().__init__(wave_params)
        # Store arguments in attributes for further use
        self.ortho_stack = ortho_stack

        self._distoshore_provider: DisToShoreProvider
        self.set_distoshore_provider(InfinityDisToShoreProvider())  # default DisToShoreProvider

        self._gravity_provider: GravityProvider
        self.set_gravity_provider(LatitudeVaryingGravityProvider())  # default GravityProvider

        # No default DeltaTimeProvider
        self._delta_time_provider: Optional[DeltaTimeProvider] = None

        # Create subtiles onto which bathymetry estimation will be done
        self.subtiles = SampledOrthoImage.build_subtiles(self.ortho_stack, nb_subtiles_max,
                                                         self.sampling_step_x,
                                                         self.sampling_step_y,
                                                         self.measure_extent)
        self._debug_samples: List[PointType] = []
        self._debug_sample = False

    @property
    def smoothing_requested(self) -> bool:
        """ :returns: True if both smoothing columns and lines parameters are non zero
        """
        return self.smoothing_columns_size != 0 and self.smoothing_lines_size != 0

    @property
    def measure_extent(self) -> MarginsType:
        """ :returns: the cartographic extent to be used for bathy estimation around a point
        """
        return (self.window_size_x / 2., self.window_size_x / 2.,
                self.window_size_y / 2., self.window_size_y / 2.)

    @property
    def selected_frames(self) -> FramesIdsType:
        """ :returns: the list of frames selected for running the estimation, or the list of all
                      the usable frames if not specified in the parameters.
        """
        selected_frames = self.selected_frames_param
        if selected_frames is None:
            selected_frames = self.ortho_stack.usable_frames
        return selected_frames

    @property
    def nb_subtiles(self) -> int:
        """ :returns: the number of subtiles
        """
        return len(self.subtiles)

    def compute_bathy(self, subtile_number: int) -> Dataset:
        """ Computes the bathymetry dataset for a given subtile.

        :param subtile_number: number of the subtile
        :returns: Subtile dataset
        """
        # Retrieve the subtile.
        subtile = self.subtiles[subtile_number]
        print(
            f'Subtile {subtile_number}: {self.ortho_stack.short_name} {subtile}')

        # Build a bathymertry estimator over the subtile and launch estimation.
        subtile_estimator = OrthoBathyEstimator(self, subtile)
        dataset = subtile_estimator.compute_bathy()

        # Build the bathymetry dataset for the subtile.
        infos = subtile_estimator.build_infos()
        infos.update(self.ortho_stack.build_infos())
        for key, value in infos.items():
            dataset.attrs[key] = value

        return dataset

# ++++++++++++++++++++++++++++ Debug support +++++++++++++++++++++++++++++
    def set_debug_area(self, bottom_left_corner: PointType, top_right_corner: PointType, decimation: int) -> None:
        """ Sets all points within rectangle defined by bottom_left_corner and top_right_corner to debug
        :param bottom_left_corner: point defining the bottom left corner of the area of interest
        :param top_right_corner: point defining the top right corner of the area of interest
        :param decimation: decimation factor for all points within the area of interest (oversize factor will lead to a single point)

        """

        x_samples = np.array([])
        y_samples = np.array([])
        for subtile in self.subtiles:
            x_samples = np.concatenate((x_samples, subtile.x_samples))
            y_samples = np.concatenate((y_samples, subtile.y_samples))
        x_samples_filtered = x_samples[np.logical_and(
            x_samples > bottom_left_corner[0], x_samples < top_right_corner[0])][::decimation]
        y_samples_filtered = y_samples[np.logical_and(
            y_samples > bottom_left_corner[1], y_samples < top_right_corner[1])][::decimation]
        list_samples = []
        for x in x_samples_filtered:
            for y in y_samples_filtered:
                list_samples.append((x, y))
        self._debug_samples = list_samples

    def set_debug_samples(self, samples: List[PointType]) -> None:
        """ Sets the list of sample points to debug

        :param samples: a list of (X,Y) tuples defining the points to debug
        """
        self._debug_samples = samples

    def set_debug(self, sample: PointType) -> None:
        """ Set or reset the debug flag for a given point depending on its presence into the set
        of points to debug.

        :param sample: The coordinate of the point for which the debug flag must be set
        """
        self._debug_sample = sample in self._debug_samples
        if self._debug_sample:
            print(f'Debugging point: X:{sample[0]} / Y:{sample[1]}')

    @property
    def debug_sample(self) -> bool:
        """ :returns: the current value of the debugging flag
        """
        return self._debug_sample

# ++++++++++++++++++++++++++++ External data providers +++++++++++++++++++

    def set_distoshore_provider(self, distoshore_provider: DisToShoreProvider) -> None:
        """ Sets the DisToShoreProvider to use with this estimator

        :param distoshore_provider: the DisToShoreProvider to use
        """
        self._distoshore_provider = distoshore_provider
        self._distoshore_provider.client_epsg_code = self.ortho_stack.epsg_code

    def get_distoshore(self, point: PointType) -> float:
        """ Provides the distance from a given point to the nearest shore.

        :param point: the point from which the distance to shore is requested.
        :returns: the distance from the point to the nearest shore (km).
        """
        return self._distoshore_provider.get_distoshore(point)

    def set_gravity_provider(self, gravity_provider: GravityProvider) -> None:
        """ Sets the GravityProvider to use with this estimator

        :param gravity_provider: the GravityProvider to use
        """
        self._gravity_provider = gravity_provider
        self._gravity_provider.client_epsg_code = self.ortho_stack.epsg_code

    def get_gravity(self, point: PointType, altitude: float = 0.) -> float:
        """ Returns the gravity at some point expressed by its X, Y and H coordinates in some SRS,
        using the gravity provider associated to this bathymetry estimator.

        :param point: a tuple containing the X and Y coordinates in the SRS set for the provider
        :param altitude: the altitude of the point in the SRS set for this provider
        :returns: the acceleration due to gravity at this point (m/s2).
        """
        return self._gravity_provider.get_gravity(point, altitude)

    def set_delta_time_provider(self, delta_time_provider: DeltaTimeProvider) -> None:
        """ Sets the DeltaTimeProvider to use with this estimator

        :param delta_time_provider: the DeltaTimeProvider to use
        """
        self._delta_time_provider = delta_time_provider
        self._delta_time_provider.client_epsg_code = self.ortho_stack.epsg_code

    def get_delta_time(self, first_frame_id: FrameIdType, second_frame_id: FrameIdType,
                       point: PointType) -> float:
        """ Returns the delta time at some point expressed by its X, Y and H coordinates in
        some SRS, using the delta time provider associated to this bathymetry estimator.

        :param first_frame_id: the id of the frame from which the duration will be counted
        :param second_frame_id: the id of the frame to which the duration will be counted
        :param point: a tuple containing the X and Y coordinates in the SRS set for the provider
        :returns: the delta time between frames at this point (s).
        :raises NoDeltaTimeProviderError: when no DeltaTimeProvider has been set for this estimator.
        """
        if self._delta_time_provider is None:
            raise NoDeltaTimeProviderError()
        return self._delta_time_provider.get_delta_time(first_frame_id, second_frame_id, point)
