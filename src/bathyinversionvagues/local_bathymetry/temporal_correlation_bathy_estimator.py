# -*- coding: utf-8 -*-
"""
Created on Wed Feb 3 10:12:00 2021

Class performing bathymetry computation using temporal correlation method

@author: erwinbergsma
         gregoirethoumyre
         degoulromain
"""
from typing import Optional, List
import numpy as np
import pandas

from ..image_processing.waves_image import WavesImage
from .local_bathy_estimator import LocalBathyEstimator
from ..image_processing.correlation_image import CorrelationImage
from ..image_processing.correlation_radon import CorrelationRadon
from ..image_processing.correlation_sinogram import CorrelationSinogram
from ..image_processing.shoresutils import cross_correlation
from ..image_processing.correlation_temporal import CorrelationTemporal


class TemporalCorrelationBathyEstimator(LocalBathyEstimator):
    def __init__(self, images_sequence: List[WavesImage], global_estimator,
                 selected_directions: Optional[np.ndarray] = None) -> None:
        """ Constructor

        :param selected_directions: the set of directions onto which the sinogram must be computed
        """
        super().__init__(images_sequence, global_estimator, selected_directions)

        self.correlation_image: CorrelationImage = None
        self.radon_transform: CorrelationRadon = None
        self.sinogram: CorrelationSinogram = None

        self._stime_series: np.array = None
        self._xx: np.array = None
        self._yy: np.array = None
        self._time_series = None
        self._partial_correlation: np.array = None
        self._projected_correlation: np.array = None
        self._distances: np.array = None
        self._angles: np.array = None

    def run(self) -> None:
        """ Run the local bathy estimator using the temporal correlation method
        """
        config = self.local_estimator_params
        params = config.TEMPORAL_METHOD

        self.create_sequence_time_series(percentage_points=params.PERCENTAGE_POINTS)
        self.compute_correlation(number_frame_shift=params.TEMPORAL_LAG,
                                 spatial_resolution=params.RESOLUTION.SPATIAL)
        self.correlation_image = CorrelationImage(pixels=self._projected_correlation,
                                                  resolution=params.RESOLUTION.SPATIAL,
                                                  tuning_ratio_size=params.TUNING.RATIO_SIZE_CORRELATION)
        self.radon_transform = CorrelationRadon(image=self.correlation_image,
                                                spatial_resolution=params.RESOLUTION.SPATIAL,
                                                time_resolution=params.RESOLUTION.TEMPORAL,
                                                temporal_lag=params.TEMPORAL_LAG,
                                                mean_filter_kernel_size=params.TUNING.MEAN_FILTER_KERNEL_SIZE_SINOGRAM)
        self.radon_transform.compute()
        self.sinogram, direction_propagation = self.radon_transform.get_sinogram_maximum_variance()

        correlation_temporal = CorrelationTemporal(angle=direction_propagation,
                                                   angles=self._angles,
                                                   distances=self._distances,
                                                   celerity=self.sinogram.celerity,
                                                   correlation_matrix=self._partial_correlation,
                                                   time_interpolation_resolution=params.RESOLUTION.TIME_INTERPOLATION,
                                                   low_frequency_ratio=params.TUNING.LOW_FREQUENCY_RATIO_TEMPORAL_RECONSTRUCTION,
                                                   high_frequency_ratio=params.TUNING.HIGH_FREQUENCY_RATIO_TEMPORAL_RECONSTRUCTION,
                                                   min_peaks_distance=params.TUNING.MIN_PEAKS_DISTANCE_PERIOD)

        waves_field_estimation = self.create_waves_field_estimation(direction_propagation,
                                                                    self.sinogram.wave_length)
        waves_field_estimation.period = correlation_temporal.period
        waves_field_estimation.celerity = self.sinogram.celerity
        self.store_estimation(waves_field_estimation)

    def create_sequence_time_series(self, percentage_points: float):
        """
        This function computes an np.array of time series.
        To do this random points are selected within the sequence of image and a temporal serie
        is included in the np.array for each selected point
        :param percentage_points (float) : percentage of selected points among all available points
        """
        if percentage_points < 0 or percentage_points > 100:
            raise ValueError("Percentage must be between 0 and 100")
        merge_array = np.dstack([image.pixels for image in self.images_sequence])
        nx, ny = self.images_sequence[0].pixels.shape
        time_series = np.reshape(merge_array, (nx * ny, -1))
        nb_random_points = round(nx * ny * percentage_points / 100)
        random_indexes = np.random.randint(0, nx * ny, size=nb_random_points)
        yy, xx = np.meshgrid(np.linspace(1, nx, nx), np.linspace(1, ny, ny))
        self._xx = xx.flatten()[random_indexes]
        self._yy = yy.flatten()[random_indexes]
        self._time_series = time_series[random_indexes, :]

    def compute_correlation(self, number_frame_shift, spatial_resolution):
        """
        This function computes the correlation of each time serie with each time serie but shifted
        of number_frame_shift frames.
        A projection is done so axes of matrix represent distances and matrix center represent the 0

        :param number_frame_shift (int): number of shifted frames to compute correlation
        :param spatial_resolution (float): spatial resolution of waveimages
        """
        self._partial_correlation = cross_correlation(self._time_series[:, number_frame_shift:],
                                                      self._time_series[:, :-number_frame_shift])

        xx = np.reshape(self._xx, (1, -1))
        yy = np.reshape(self._yy, (1, -1))
        self._distances = np.sqrt(np.square((xx - xx.T)) + np.square((yy - yy.T)))
        xrawipool_ik_dist = np.tile(xx, (len(xx), 1)) - np.tile(xx.T, (1, len(xx)))
        yrawipool_ik_dist = np.tile(yy, (len(yy), 1)) - np.tile(yy.T, (1, len(yy)))
        self._angles = np.arctan2(xrawipool_ik_dist, yrawipool_ik_dist).T  # angles are in radian

        xr = np.round(self._distances * np.cos(self._angles) / spatial_resolution)
        xr = np.array(xr - np.min(xr), dtype=int).T

        yr = np.round(self._distances * np.sin(self._angles) / spatial_resolution)
        yr = np.array(yr - np.min(yr), dtype=int).T

        xr_s = pandas.Series(xr.flatten())
        yr_s = pandas.Series(yr.flatten())
        values_s = pandas.Series(self._partial_correlation.flatten())

        # if two correlation values have same xr and yr mean of these values is taken
        dataframe = pandas.DataFrame({'xr': xr_s, 'yr': yr_s, 'values': values_s})
        d = dataframe.groupby(by=['xr', 'yr']).mean().reset_index()
        values = np.array(d['values'])
        xr = np.array(d['xr'])
        yr = np.array(d['yr'])

        projected_matrix = np.nanmean(self._partial_correlation) * np.ones(
            (np.max(xr) + 1, np.max(yr) + 1))
        projected_matrix[xr, yr] = values
        self._projected_correlation = projected_matrix
