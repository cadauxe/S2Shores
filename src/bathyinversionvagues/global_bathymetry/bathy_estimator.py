# -*- coding: utf-8 -*-
""" Definition of the BathyEstimator abstract class

:author: GIROS Alain
:created: 17/05/2021
"""
from abc import ABC
from pathlib import Path
from typing import List, Optional, Dict, Union, Any  # @NoMove


import xarray as xr  # @NoMove
from xarray import Dataset  # @NoMove


import numpy as np

from ..data_providers.delta_time_provider import DeltaTimeProvider, NoDeltaTimeProviderError
from ..data_providers.dis_to_shore_provider import (InfinityDisToShoreProvider, DisToShoreProvider,
                                                    NetCDFDisToShoreProvider, GeotiffDisToShoreProvider)
from ..data_providers.gravity_provider import (LatitudeVaryingGravityProvider, GravityProvider,
                                               ConstantGravityProvider)
from ..data_providers.roi_provider import (RoiProvider, VectorFileRoiProvider)
from ..image.image_geometry_types import MarginsType, PointType
from ..image.ortho_stack import OrthoStack, FrameIdType, FramesIdsType
from ..image.sampled_ortho_image import SampledOrthoImage
from .bathy_estimator_parameters import BathyEstimatorParameters
from .ortho_bathy_estimator import OrthoBathyEstimator


# TODO: create an abstract BathyEstimatorProviders class to host the different providers
class BathyEstimator(ABC, BathyEstimatorParameters):
    """ Management of bathymetry computation and parameters on a single product. Computation
    is split in several cartographic tiles, which must be run separately, either in parallel or
    sequentially.
    """

    def __init__(self, ortho_stack: OrthoStack, wave_params: Dict[str, Any], output_dir: Path,
                 nb_subtiles_max: int = 1) -> None:
        """Create a BathyEstimator object and set necessary informations

        :param ortho_stack: the orthorectified stack onto which bathymetry must be estimated.
        :param wave_params: parameters for the global and local bathymetry estimators
        :param output_dir: path to the directory where the netCDF bathy file will be written.
        :param nb_subtiles_max: Nb of subtiles for bathymetry estimation
        """
        super().__init__(wave_params)
        # Store arguments in attributes for further use
        self._ortho_stack = ortho_stack
        self._output_dir = output_dir

        self._distoshore_provider: DisToShoreProvider
        # set InfinityDisToShoreProvider as default DisToShoreProvider
        self.set_distoshore_provider(provider_info=InfinityDisToShoreProvider())

        self._gravity_provider: GravityProvider
        # set LatitudeVaryingGravityProvider as default GravityProvider
        self.set_gravity_provider(provider_info=LatitudeVaryingGravityProvider())

        # No default DeltaTimeProvider
        self._delta_time_provider: Optional[DeltaTimeProvider] = None

        # No default RoiProvider
        self._roi_provider: Optional[RoiProvider] = None
        self._limit_to_roi = False

        # Create subtiles onto which bathymetry estimation will be done
        self._nb_subtiles_max = nb_subtiles_max
        self.subtiles: List[SampledOrthoImage]

        # Init debugging points handling
        self._debug_path: Optional[Path] = None
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
            selected_frames = self._ortho_stack.usable_frames
        return selected_frames

    @property
    def nb_subtiles(self) -> int:
        """ :returns: the number of subtiles
        """
        return len(self.subtiles)

    def create_subtiles(self) -> None:
        """ Warmup of the bathy estimator by creating the processing subtiles
        """
        roi = None
        if self._roi_provider is not None and self._limit_to_roi:
            roi = self._roi_provider.bounding_box(0.1)
        self.subtiles = SampledOrthoImage.build_subtiles(self._ortho_stack,
                                                         self._nb_subtiles_max,
                                                         self.sampling_step_x,
                                                         self.sampling_step_y,
                                                         self.measure_extent,
                                                         roi=roi)

    def compute_bathy_for_subtile(self, subtile_number: int) -> Dataset:
        """ Computes the bathymetry dataset for a given subtile.

        :param subtile_number: number of the subtile
        :returns: Subtile dataset
        """
        # Retrieve the subtile.
        subtile = self.subtiles[subtile_number]
        print(f'Subtile {subtile_number}: {self._ortho_stack.short_name} {subtile}')

        # Build a bathymertry estimator over the subtile and launch estimation.
        subtile_estimator = OrthoBathyEstimator(self, subtile)
        dataset = subtile_estimator.compute_bathy()

        # Build the bathymetry dataset for the subtile.
        infos = self.build_infos()
        infos.update(self._ortho_stack.build_infos())
        for key, value in infos.items():
            dataset.attrs[key] = value

        # We return the dataset instead of storing it in the instance, for multiprocessing reasons.
        return dataset

    def merge_subtiles(self, bathy_subtiles: List[Dataset]) -> None:
        """Merge all the subtiles datasets in memory into a single one in a netCDF file

        :param bathy_subtiles: Subtiles datasets
        """
        merged_bathy = xr.combine_by_coords(bathy_subtiles)
        product_name = self._ortho_stack.full_name
        netcdf_output_path = (self._output_dir / product_name).with_suffix('.nc')
        merged_bathy.to_netcdf(path=netcdf_output_path, format='NETCDF4')

    def build_infos(self) -> Dict[str, str]:
        """ :returns: a dictionary of metadata describing this estimator
        """

        title = 'Wave parameters and raw bathymetry derived from satellite imagery.'
        title += ' No tidal vertical adjustment.'
        infos = {'title': title,
                 'institution': 'CNES-LEGOS'}

        # metadata from the parameters
        infos['waveEstimationMethod'] = self.local_estimator_code
        infos['ChainVersions'] = self.chains_versions
        infos['Resolution X'] = str(self.sampling_step_x)
        infos['Resolution Y'] = str(self.sampling_step_y)

        return infos

# ++++++++++++++++++++++++++++ Debug support +++++++++++++++++++++++++++++
    @property
    def debug_path(self) -> Optional[Path]:
        """ :returns: path to a directory where debugging info can be written.
        """
        return self._debug_path

    @debug_path.setter
    def debug_path(self, path: Path) -> None:
        self._debug_path = path

    def set_debug_area(self, bottom_left_corner: PointType, top_right_corner: PointType,
                       decimation: int) -> None:
        """ Sets all points within rectangle defined by bottom_left_corner and top_right_corner to
        debug

        :param bottom_left_corner: point defining the bottom left corner of the area of interest
        :param top_right_corner: point defining the top right corner of the area of interest
        :param decimation: decimation factor for all points within the area of interest
                           (oversize factor will lead to a single point)

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
        for x_coord in x_samples_filtered:
            for y_coord in y_samples_filtered:
                list_samples.append((x_coord, y_coord))
        self._debug_samples = list_samples

    def set_debug_samples(self, samples: List[PointType]) -> None:
        """ Sets the list of sample points to debug

        :param samples: a list of (X,Y) tuples defining the points to debug
        """
        self._debug_samples = samples

    def set_debug_flag(self, sample: PointType) -> None:
        """ Set or reset the debug flag for a given point depending on its presence into the set
        of points to debug.

        :param sample: The coordinate of the point for which the debug flag must be set
        """
        self._debug_sample = sample in self._debug_samples

    @property
    def debug_sample(self) -> bool:
        """ :returns: the current value of the debugging flag
        """
        return self._debug_sample

# ++++++++++++++++++++++++++++ External data providers +++++++++++++++++++
    def set_distoshore_provider(
            self, provider_info: Optional[Union[Path, DisToShoreProvider]] = None) -> None:
        """ Sets the DisToShoreProvider to use with this estimator

        :param provider_info: Either the DisToShoreProvider to use or a path to a netCDF or Geotiff file
                           assuming a geographic NetCDF or Geotiff format.
        """
        if isinstance(provider_info, DisToShoreProvider):
            distoshore_provider = provider_info
        elif isinstance(provider_info, Path):
            if (Path(provider_info).suffix.lower() == '.nc'):
                distoshore_provider = NetCDFDisToShoreProvider(provider_info, 4326,
                                                               x_axis_label='lon',
                                                               y_axis_label='lat')
            elif (Path(provider_info).suffix.lower() == '.tif'):
                distoshore_provider = GeotiffDisToShoreProvider(provider_info, 4326,
                                                                x_axis_label='lon',
                                                                y_axis_label='lat')
        else:
            # None or some other type, keep the current provider
            distoshore_provider = self._distoshore_provider

        # Set private attribute.
        self._distoshore_provider = distoshore_provider
        if self._distoshore_provider is not None:
            self._distoshore_provider.client_epsg_code = self._ortho_stack.epsg_code

    def get_distoshore(self, point: PointType) -> float:
        """ Provides the distance from a given point to the nearest shore.

        :param point: the point from which the distance to shore is requested.
        :returns: the distance from the point to the nearest shore (km).
        """
        return self._distoshore_provider.get_distoshore(point)

    def set_roi_provider(self, provider_info: Optional[Union[Path, RoiProvider]] = None,
                         limit_to_roi: bool = False) -> None:
        """ Sets the RoiProvider to use with this estimator

        :param provider_info: Either the RoiProvider to use or a path to a vector file containing
                              the ROI or None if no provider change.
        :param limit_to_roi: if True, the produced bathymetry will be limited to a bounding box
                             enclosing the Roi with some margins.
        """
        roi_provider: Optional[RoiProvider]
        if isinstance(provider_info, RoiProvider):
            roi_provider = provider_info
        elif isinstance(provider_info, Path):
            roi_provider = VectorFileRoiProvider(provider_info)
        else:
            # None or some other type, keep the current provider
            roi_provider = self._roi_provider

        # Set private attribute.
        self._roi_provider = roi_provider
        if self._roi_provider is not None:
            self._roi_provider.client_epsg_code = self._ortho_stack.epsg_code
            self._limit_to_roi = limit_to_roi

    def is_inside_roi(self, point: PointType) -> bool:
        """ Test if a point is inside the ROI

        :param point: the point to test
        :returns: True if the point lies inside the ROI.
        """
        if self._roi_provider is None:
            return True
        return self._roi_provider.contains(point)

    def set_gravity_provider(self,
                             provider_info: Optional[Union[str, GravityProvider]] = None) -> None:
        """ Sets the GravityProvider to use with this estimator .

        :param provider_info: an instance of GravityProvider or the name of a well known gravity
                              provider to use. If None the current provider is left unchanged.
        :raises ValueError: when the gravity provider name is unknown
        """
        if isinstance(provider_info, GravityProvider):
            gravity_provider = provider_info
        elif isinstance(provider_info, str):
            if provider_info.upper() not in ['CONSTANT', 'LATITUDE_VARYING']:
                raise ValueError('Gravity provider type unknown : ', provider_info)
            # No need to set LatitudeVaryingGravityProvider as it is the BathyEstimator default.
            if provider_info.upper() == 'CONSTANT':
                gravity_provider = ConstantGravityProvider()
            else:
                gravity_provider = LatitudeVaryingGravityProvider()
        else:
            # None or some other type, keep the current provider
            gravity_provider = self._gravity_provider

        # Set private attribute.
        self._gravity_provider = gravity_provider
        if self._gravity_provider is not None:
            self._gravity_provider.client_epsg_code = self._ortho_stack.epsg_code

    def get_gravity(self, point: PointType, altitude: float = 0.) -> float:
        """ Returns the gravity at some point expressed by its X, Y and H coordinates in some SRS,
        using the gravity provider associated to this bathymetry estimator.

        :param point: a tuple containing the X and Y coordinates in the SRS set for the provider
        :param altitude: the altitude of the point in the SRS set for this provider
        :returns: the acceleration due to gravity at this point (m/s2).
        """
        return self._gravity_provider.get_gravity(point, altitude)

    def set_delta_time_provider(
            self, provider_info: Optional[Union[Path, DeltaTimeProvider]] = None) -> None:
        """ Sets the DeltaTimeProvider to use with this estimator.

        :param provider_info: Either the DeltaTimeProvider to use or a path to a file or a
                              directory to ba used by the associated OrthoStack to build its
                              provider, or None to leave the provider unchanged.
        """
        delta_time_provider: Optional[DeltaTimeProvider]
        if isinstance(provider_info, DeltaTimeProvider):
            delta_time_provider = provider_info
        else:
            delta_time_provider = self._ortho_stack.create_delta_time_provider(provider_info)

        # Set private attribute.
        self._delta_time_provider = delta_time_provider
        if self._delta_time_provider is not None:
            self._delta_time_provider.client_epsg_code = self._ortho_stack.epsg_code

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
