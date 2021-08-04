# -*- coding: utf-8 -*-
"""
Created on Wed Feb 3 10:12:00 2021

Class performing bathymetry computation using temporal correlation method

@author: erwinbergsma
         gregoirethoumyre
         degoulromain
"""
import numpy as np

from ..image_processing.shoresutils import (fft_filtering, compute_sinogram,
                                            compute_celerity, correlation_tuning,
                                            sinogram_tuning, compute_wave_length, compute_period,
                                            temporal_reconstruction,
                                            temporal_reconstruction_tuning,
                                            create_sequence_time_series_temporal,
                                            cartesian_projection, compute_temporal_correlation,
                                            )

from ..image_processing.waves_image import WavesImage
from .local_bathy_estimator import LocalBathyEstimator
from ..image_processing.correlation_image import CorrelationImage
from ..image_processing.correlation_radon import CorrelationRadon
from ..image_processing.correlation_sinogram import CorrelationSinogram
from typing import Optional, List

import matplotlib.pyplot as plt


class TemporalCorrelationBathyEstimator(LocalBathyEstimator):
    def __init__(self, images_sequence: List[WavesImage], global_estimator,
                 selected_directions: Optional[np.ndarray] = None) -> None:
        """ Constructor

        :param selected_directions: the set of directions onto which the sinogram must be computed
        """
        super().__init__(images_sequence, global_estimator, selected_directions)

        self.correlation_cartesian_matrix: CorrelationImage = None
        self.radon_transform: CorrelationRadon = None


    def run(self) -> None:
        """ Run the local bathy estimator using the temporal correlation method

        """
        config = self.local_estimator_params
        params = config.TEMPORAL_METHOD

        # FIXME: temporary adaptor before getting rid of stacked np.ndarrays.
        Im = np.dstack([image.pixels for image in self.images_sequence])

        # TODO : Refractoring with classes
        ##############################################################################################################
        stime_series, xx, yy = create_sequence_time_series_temporal(Im=Im,
                                                                    percentage_points=params.PERCENTAGE_POINTS)
        corr = compute_temporal_correlation(sequence_thumbnail=stime_series,
                                            number_frame_shift=params.TEMPORAL_LAG)
        corr_car, distances, angles = cartesian_projection(corr_matrix=corr, xx=xx, yy=yy,
                                                           spatial_resolution=params.RESOLUTION.SPATIAL)
        ##############################################################################################################
        self.correlation_cartesian_matrix = CorrelationImage(pixels=corr_car, resolution=params.RESOLUTION.SPATIAL,
                                                             tuning_ratio_size=params.TUNING.RATIO_SIZE_CORRELATION)
        self.radon_transform = CorrelationRadon(image=self.correlation_cartesian_matrix,
                                                spatial_resolution=params.RESOLUTION.SPATIAL,
                                                time_resolution=params.RESOLUTION.TEMPORAL,
                                                temporal_lag=params.TEMPORAL_LAG,
                                                mean_filter_kernel_size=params.TUNING.MEAN_FILTER_KERNEL_SIZE_SINOGRAM)
        self.radon_transform.compute()
        sinograw_max_var, direction_propagation = self.radon_transform.get_sinogram_maximum_variance()
        ##############################################################################################################
        SS = temporal_reconstruction(angle=direction_propagation, angles=np.degrees(angles), distances=distances,
                                     celerity=sinograw_max_var.celerity,
                                     correlation_matrix=corr,
                                     time_interpolation_resolution=params.RESOLUTION.TIME_INTERPOLATION)
        SS_filtered = temporal_reconstruction_tuning(SS,
                                                     time_interpolation_resolution=params.RESOLUTION.TIME_INTERPOLATION,
                                                     low_frequency_ratio=params.TUNING.LOW_FREQUENCY_RATIO_TEMPORAL_RECONSTRUCTION,
                                                     high_frequency_ratio=params.TUNING.HIGH_FREQUENCY_RATIO_TEMPORAL_RECONSTRUCTION)
        T, peaks_max = compute_period(SS_filtered=SS_filtered,
                                      min_peaks_distance=params.TUNING.MIN_PEAKS_DISTANCE_PERIOD)

        ##############################################################################################################
        waves_field_estimation = self.create_waves_field_estimation(direction_propagation, sinograw_max_var.wave_length)
        waves_field_estimation.period = T
        waves_field_estimation.celerity = sinograw_max_var.celerity

        self.store_estimation(waves_field_estimation)
