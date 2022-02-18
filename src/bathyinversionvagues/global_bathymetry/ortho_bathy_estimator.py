# -*- coding: utf-8 -*-
""" Definition of the OrthoBathyEstimator class

:author: GIROS Alain
:created: 05/05/2021
"""
import time
import warnings

from ..data_model.estimated_bathy import EstimatedBathy
from ..data_model.waves_fields_estimations import WavesFieldsEstimations
from ..data_providers.delta_time_provider import NoDeltaTimeValueError
from ..image.image_geometry_types import PointType
from ..image.sampled_ortho_image import SampledOrthoImage
from ..image_processing.waves_image import WavesImage
from ..local_bathymetry.local_bathy_estimator_factory import local_bathy_estimator_factory
from ..waves_exceptions import WavesException


from typing import List, TYPE_CHECKING  # @NoMove

from xarray import Dataset  # @NoMove


if TYPE_CHECKING:
    from .bathy_estimator import BathyEstimator  # @UnusedImport


# TODO: create a WavesImageSequence class holding a list of WavesImage
# TODO: Make this class inherit from BathyEstimator ?
class OrthoBathyEstimator:
    """ This class implements the computation of bathymetry over a sampled orthorectifed image.
    """

    def __init__(self, estimator: 'BathyEstimator', sampled_ortho: SampledOrthoImage) -> None:
        """ Constructor

        :param estimator: the parent estimator of this estimator
        :param sampled_ortho: the image onto which the bathy estimation must be done
        """
        self.sampled_ortho = sampled_ortho
        self.parent_estimator = estimator

    def compute_bathy(self) -> Dataset:
        """ Computes the bathymetry dataset for the samples belonging to a given subtile.

        :return: Estimated bathymetry dataset
        """

        start_load = time.time()
        # nbkeep shall be understood as a filtering in terms of the number of proposed samples.
        # Will disappear when true Waves Fields will be identified and implemented.
        nb_keep = self.parent_estimator.nb_max_waves_fields

        estimated_bathy = EstimatedBathy(self.sampled_ortho.x_samples, self.sampled_ortho.y_samples,
                                         self.sampled_ortho.ortho_stack.acquisition_time)

        # subtile reading
        sub_tile_images = [self.sampled_ortho.read_pixels(frame_id) for
                           frame_id in self.parent_estimator.selected_frames]
        print(f'Loading time: {time.time() - start_load:.2f} s')

        start = time.time()
        computed_points = 0
        for x_sample in self.sampled_ortho.x_samples:
            for y_sample in self.sampled_ortho.y_samples:
                estimation_point = (x_sample, y_sample)
                self.parent_estimator.set_debug_flag(estimation_point)
                bathy_estimations = self._run_local_bathy_estimator(sub_tile_images,
                                                                    estimation_point)
                if bathy_estimations.distance_to_shore > 0 and bathy_estimations.inside_roi:
                    computed_points += 1

                # Store bathymetry sample estimations
                estimated_bathy.store_estimations(x_sample, y_sample, bathy_estimations)

        total_points = self.sampled_ortho.nb_samples
        comput_time = time.time() - start
        print(f'Computed {computed_points}/{total_points} points in: {comput_time:.2f} s')

        return estimated_bathy.build_dataset(self.parent_estimator.layers_type, nb_keep)

    def _run_local_bathy_estimator(self, sub_tile_images: List[WavesImage],
                                   estimation_point: PointType) -> WavesFieldsEstimations:
        distance = self.parent_estimator.get_distoshore(estimation_point)
        gravity = self.parent_estimator.get_gravity(estimation_point, 0.)
        inside_roi = self.parent_estimator.is_inside_roi(estimation_point)
        bathy_estimations = WavesFieldsEstimations(estimation_point, gravity, distance, inside_roi)
        # do not compute on land
        # FIXME: distance to shore test should take into account windows sizes
        if distance > 0 and inside_roi:
            # computes the bathymetry at the specified position
            try:
                images_sequence = self._create_images_sequence(sub_tile_images,
                                                               estimation_point)
                # TODO: use selected_directions argument
                local_bathy_estimator = local_bathy_estimator_factory(images_sequence,
                                                                      self.parent_estimator,
                                                                      bathy_estimations)
                local_bathy_estimator.run()
                local_bathy_estimator.validate_waves_fields()
                local_bathy_estimator.sort_waves_fields()
                if self.parent_estimator.debug_sample:
                    print(f'estimations after sorting :')
                    print(local_bathy_estimator.waves_fields_estimations)
            except NoDeltaTimeValueError:
                bathy_estimations.delta_time_available = False
                bathy_estimations.clear()
            except WavesException as excp:
                warn_msg = f'Unable to estimate bathymetry: {str(excp)}'
                warnings.warn(warn_msg)
                bathy_estimations.clear()
        return bathy_estimations

    def _create_images_sequence(self, sub_tile_images: List[WavesImage],
                                estimation_point: PointType) -> List[WavesImage]:

        window = self.sampled_ortho.window_extent(estimation_point)

        # Create the sequence of WavesImages (to be used by ALL estimators)
        images_sequence: List[WavesImage] = []
        for index, frame_id in enumerate(self.parent_estimator.selected_frames):
            # TODO: make a method in WavesImage to create an excerpt ?
            pixels = sub_tile_images[index].pixels
            window_image = WavesImage(pixels[window[0]:window[1] + 1, window[2]:window[3] + 1],
                                      sub_tile_images[index].resolution)
            images_sequence.append(window_image)
            if self.parent_estimator.debug_sample:
                print(f'Subtile shape {sub_tile_images[index].pixels.shape}')
                print(f'Window in ortho image coordinate: {window}')
                print(f'--{frame_id} imagette {window_image.pixels.shape}:')
                print(window_image.pixels)
        return images_sequence
