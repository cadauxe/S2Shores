# -*- coding: utf-8 -*-
""" Definition of the OrthoBathyEstimator class

:author: GIROS Alain
:created: 05/05/2021
"""
import time

from typing import Dict, TYPE_CHECKING

import numpy as np  # @NoMove
from xarray import Dataset  # @NoMove
import xarray as xr  # @NoMove

from ..image.sampled_ortho_image import SampledOrthoImage
from ..local_bathymetry_estimation import wave_parameters_and_bathy_estimation

from .estimated_bathy import EstimatedBathy


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

        # images reading
        sub_image_ref = self.sampled_ortho.read_pixels(self.parent_estimator.ref_band_id)
        sub_image_sec = self.sampled_ortho.read_pixels(self.parent_estimator.sec_band_id)
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
                    # computes the window in the image space
                    window = self.sampled_ortho.window_extent((x_sample, y_sample))

                    subimageref = sub_image_ref[window[0]:window[1] + 1, window[2]:window[3] + 1]
                    subimagesec = sub_image_sec[window[0]:window[1] + 1, window[2]:window[3] + 1]
                    if self.parent_estimator.debug_sample:
                        print(f'Subtile shape {sub_image_ref.shape}')
                        print(f'Window in ortho image coordinate: {window}')
                        print(f'------- {self.parent_estimator.ref_band_id} reference imagette:')
                        print(subimageref)
                        print(f'Mean ref: {np.mean(subimageref)}')
                        print(f'------- {self.parent_estimator.sec_band_id} secondary imagette:')
                        print(subimagesec)
                        print(f'Mean sec: {np.mean(subimagesec)}')
                    # Bathymetry computation
                    images_sequence = np.dstack((subimageref, subimagesec))
                    wave_bathy_point = wave_parameters_and_bathy_estimation(images_sequence,
                                                                            self.parent_estimator)

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
