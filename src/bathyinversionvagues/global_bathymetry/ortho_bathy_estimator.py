# -*- coding: utf-8 -*-
""" Definition of the OrthoBathyEstimator class

:author: GIROS Alain
:created: 05/05/2021
"""
import time
from typing import Dict, List, TYPE_CHECKING

from ..image.sampled_ortho_image import SampledOrthoImage
from ..image_processing.waves_image import WavesImage
from ..local_bathymetry.local_bathymetry_estimation import wave_parameters_and_bathy_estimation
from .estimated_bathy import EstimatedBathy


import numpy as np  # @NoMove
from xarray import Dataset  # @NoMove
import xarray as xr  # @NoMove


if TYPE_CHECKING:
    from .bathy_estimator import BathyEstimator  # @UnusedImport


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
        nb_keep = self.parent_estimator.waveparams.NKEEP
        layers_type = self.parent_estimator.waveparams.LAYERS_TYPE

        estimated_bathy = EstimatedBathy(self.sampled_ortho.x_samples, self.sampled_ortho.y_samples,
                                         self.sampled_ortho.image.acquisition_time, nb_keep)
        # distance to shore (km)
        distoshore = xr.open_dataset(self.parent_estimator.distoshore_file_path)

        # subtile reading
        sub_tile_images: List[np.ndarray] = []
        for band_id in self.parent_estimator.bands_identifiers:
            sub_tile_images.append(self.sampled_ortho.read_pixels(band_id))
        print(f'Loading time: {time.time() - start_load:.2f} s')

        start = time.time()
        in_water_points = 0
        for i, x_sample in enumerate(self.sampled_ortho.x_samples):
            for j, y_sample in enumerate(self.sampled_ortho.y_samples):
                self.parent_estimator.set_debug((x_sample, y_sample))
                # distance to shore loading
                # FIXME: following line needed to deal with upside down distoshore files
                corrected_yp = self.sampled_ortho.image.upper_left_y + \
                    self.sampled_ortho.image.lower_right_y - y_sample
                distance = distoshore.disToShore.sel(y=corrected_yp, x=x_sample,
                                                     method='nearest')  # km

                # do not compute on land
                # FIXME: distance to shore test should take into account windows sizes
                if distance > 0:
                    in_water_points += 1
                    # computes the bathymetry at the specified position
                    wave_bathy_point = self.compute_local_bathy(sub_tile_images, x_sample, y_sample)
                else:
                    wave_bathy_point = estimated_bathy.empty_sample

                # Store bathymetry sample
                # FIXME: distance is a DataArray xarray instance
                wave_bathy_point['distoshore'] = distance
                estimated_bathy.store_sample(i, j, wave_bathy_point)

        total_points = self.sampled_ortho.nb_samples
        comput_time = time.time() - start
        print(f'Computed {in_water_points}/{total_points} points in: {comput_time:.2f} s')

        return estimated_bathy.build_dataset(layers_type)

    def compute_local_bathy(self, sub_tile_images, x_sample, y_sample):

        window = self.sampled_ortho.window_extent((x_sample, y_sample))
        # TODO: Link WavesImage to OrthoImage and use resolution from it?
        resolution = self.parent_estimator.waveparams.DX  # in meter
        # Create the sequence of WavesImages (to be used by ALL estimators)
        if self.parent_estimator.smoothing_requested:
            smoothing = (self.parent_estimator.smoothing_lines_size,
                         self.parent_estimator.smoothing_columns_size)
        else:
            smoothing = None

        images_sequence: List[WavesImage] = []
        for index, band_id in enumerate(self.parent_estimator.bands_identifiers):
            window_image = WavesImage(sub_tile_images[index][window[0]:window[1] + 1,
                                                             window[2]:window[3] + 1],
                                      resolution, smoothing=smoothing)
            images_sequence.append(window_image)
            if self.parent_estimator.debug_sample:
                print(f'Subtile shape {sub_tile_images[index].shape}')
                print(f'Window in ortho image coordinate: {window}')
                print(f'--{band_id} imagette {window_image.pixels.shape}:')
                print(window_image.pixels)

        # Local bathymetry computation
        wave_bathy_point = wave_parameters_and_bathy_estimation(images_sequence,
                                                                self.parent_estimator)
        return wave_bathy_point

    def build_infos(self) -> Dict[str, str]:
        """ :returns: a dictionary of metadata describing this estimator
        """

        title = 'Wave parameters and raw bathymetry derived from satellite imagery.'
        title += ' No tidal vertical adjustment.'
        infos = {'title': title,
                 'institution': 'CNES-LEGOS'}

        # metadata from the parameters
        infos['waveEstimationMethod'] = self.parent_estimator.waveparams.WAVE_EST_METHOD

        return infos
