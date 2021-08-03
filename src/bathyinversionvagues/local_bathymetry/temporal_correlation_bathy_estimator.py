# -*- coding: utf-8 -*-
"""
Created on Wed Feb 3 10:12:00 2021

Module containing all wave parameters estimation methods

@author: erwinbergsma
         gregoirethoumyre
         degoulromain
"""
import numpy as np

from ..image_processing.shoresutils import (compute_period,
                                            )

from ..image_processing.waves_image import WavesImage
from .local_bathy_estimator import LocalBathyEstimator
from ..image_processing.correlation_image import CorrelationImage
from ..image_processing.correlation_radon import CorrelationRadon
from ..image_processing.shoresutils import cross_correlation
from scipy.interpolate import interp1d
from typing import Optional, List
import pandas
import scipy


class TemporalCorrelationBathyEstimator(LocalBathyEstimator):
    def __init__(self, images_sequence: List[WavesImage], global_estimator,
                 selected_directions: Optional[np.ndarray] = None) -> None:
        """ Constructor

        :param selected_directions: the set of directions onto which the sinogram must be computed
        """
        super().__init__(images_sequence, global_estimator, selected_directions)

        self.correlation_cartesian_matrix: CorrelationImage = None
        self.radon_transform: CorrelationRadon = None
        self.celerity: float = None
        self.wave_length: float = None
        self.period: float = None
        self.direction_propagation: float = None

    def run(self) -> None:
        """ Run the local bathy estimator using the temporal correlation method
        """
        config = self.local_estimator_params
        params = config.TEMPORAL_METHOD

        stime_series, xx, yy = self.create_sequence_time_series(percentage_points=params.PERCENTAGE_POINTS)
        corr = self.compute_correlation(sequence_time_series=stime_series,
                                        number_frame_shift=params.TEMPORAL_LAG)
        corr_car, distances, angles = self.cartesian_projection(corr_matrix=corr, xx=xx, yy=yy,
                                                                spatial_resolution=params.RESOLUTION.SPATIAL)
        self.correlation_cartesian_matrix = CorrelationImage(pixels=corr_car, resolution=params.RESOLUTION.SPATIAL,
                                                             tuning_ratio_size=params.TUNING.RATIO_SIZE_CORRELATION)
        self.radon_transform = CorrelationRadon(image=self.correlation_cartesian_matrix,
                                                spatial_resolution=params.RESOLUTION.SPATIAL,
                                                time_resolution=params.RESOLUTION.TEMPORAL,
                                                temporal_lag=params.TEMPORAL_LAG,
                                                mean_filter_kernel_size=params.TUNING.MEAN_FILTER_KERNEL_SIZE_SINOGRAM)
        self.radon_transform.compute()
        sinogram_max_var, self.direction_propagation = self.radon_transform.get_sinogram_maximum_variance()
        self.celerity = sinogram_max_var.celerity
        self.wave_length = sinogram_max_var.wave_length
        SS = self.temporal_reconstruction(angle=self.direction_propagation, angles=np.degrees(angles),
                                          distances=distances,
                                          celerity=sinogram_max_var.celerity,
                                          correlation_matrix=corr,
                                          time_interpolation_resolution=params.RESOLUTION.TIME_INTERPOLATION)
        SS_filtered = self.temporal_reconstruction_tuning(SS,
                                                          time_interpolation_resolution=params.RESOLUTION.TIME_INTERPOLATION,
                                                          low_frequency_ratio=params.TUNING.LOW_FREQUENCY_RATIO_TEMPORAL_RECONSTRUCTION,
                                                          high_frequency_ratio=params.TUNING.HIGH_FREQUENCY_RATIO_TEMPORAL_RECONSTRUCTION)
        self.period, peaks_max = self.compute_period(SS_filtered=SS_filtered,
                                                     min_peaks_distance=params.TUNING.MIN_PEAKS_DISTANCE_PERIOD)

        waves_field_estimation = self.create_waves_field_estimation(self.direction_propagation,
                                                                    sinogram_max_var.wave_length)
        waves_field_estimation.period = self.period
        waves_field_estimation.celerity = sinogram_max_var.celerity

        self.store_estimation(waves_field_estimation)

    def create_sequence_time_series(self, percentage_points: float):
        """
            This function takes a sequence of images, filters (passband) sequence and takes random time series within the thumbnail.
            Thumbnail is flatten on the first axis so sequence_thumbnail returned is shape (number_of_random_points,number_of_frames)
            :param Im (numpy array of size (number_of_lines,number_of_column,number_of_frames)) : sequence of thumbnails
            :param percentage_points (int) : percentage of points taken within the thumbnail
            :param spatial_resolution (int) : percentage of points taken within the thumbnail
            :param fft_T_max (int) : Max wave period to be allowed
            :param fft_T_min (int) : Min wave period to be allowed
            :return sequence_time_series (numpy array of size (number_of_random_points,number_of_frames)) : array of random time series
                    xx (flatten numpy array of size number_of_random_points) : list x value of random points
                    yy (flatten numpy array of size number_of_random_points) : list y value of random points
            """
        if percentage_points < 0 or percentage_points > 100:
            raise ValueError("Percentage must be between 0 and 100")
        merge_array = np.dstack([image.pixels for image in self.images_sequence])
        nx, ny = self.images_sequence[0].pixels.shape
        time_series = np.reshape(merge_array, (nx * ny, -1))
        nb_random_points = round(nx * ny * percentage_points / 100)
        random_indexes = np.random.randint(0, nx * ny, size=nb_random_points)
        yy, xx = np.meshgrid(np.linspace(1, nx, nx), np.linspace(1, ny, ny))
        xx = xx.flatten()
        yy = yy.flatten()
        return time_series[random_indexes, :], xx[random_indexes], yy[random_indexes]

    def compute_correlation(self, sequence_time_series, number_frame_shift):
        """
            This function computes the correlation of each time serie of sequence_thumbnail with each
            time serie of sequence_thumbnail but shifted of number_frame_shift frames

            :param np.ndarray sequence_thumbnail: (number_of_time_series, number_of_frames)
                                                  video of waves
            :param int number_frame_shift: number of shifted frames
            :return np.ndarray corr: (number_of_time_series,number_of_time_series)
                                     cross correlation of time series
            """
        corr = cross_correlation(sequence_time_series[:, number_frame_shift:],
                                 sequence_time_series[:, :-number_frame_shift])
        return corr

    def cartesian_projection(self, corr_matrix, xx, yy, spatial_resolution):
        """
            This function does cartesian projection of correlation matrix.
            xx and yy are list of x and y coordinates on which values in matrix are projected
            This function is meant to be used with function create_sequence_time_series
            :param corr_matrix (numpy array) : entry correlation matrix
            :param xx (numpy array) : list random x computed with function create_sequence_time_series
            :param yy (numpy array) : list random y computed with function create_sequence_time_series
            :param spatial_resolution (float) : spatial resolution
            :return projected_matrix (numpy array) : cartesian matrix, lines & column are meters
                    euclidean distance (numpy array) : pairwise distance between each random points of (xx,yy)
                    angles (numpy array) : angles between each random points of (xx,yy)
        """
        xx = np.reshape(xx, (1, -1))
        yy = np.reshape(yy, (1, -1))

        euclidean_distance = np.sqrt(np.square((xx - xx.T)) + np.square((yy - yy.T)))
        xrawipool_ik_dist = np.tile(xx, (len(xx), 1)) - np.tile(xx.T, (1, len(xx)))
        yrawipool_ik_dist = np.tile(yy, (len(yy), 1)) - np.tile(yy.T, (1, len(yy)))
        angles = np.arctan2(xrawipool_ik_dist, yrawipool_ik_dist).T  # angles are in radian
        xr = np.round(euclidean_distance * np.cos(angles) / spatial_resolution)
        xr = np.array(xr - np.min(xr), dtype=int).T

        yr = np.round(euclidean_distance * np.sin(angles) / spatial_resolution)
        yr = np.array(yr - np.min(yr), dtype=int).T

        xr_s = pandas.Series(xr.flatten())
        yr_s = pandas.Series(yr.flatten())
        values_s = pandas.Series(corr_matrix.flatten())

        # if two correlation values have same xr and yr mean of these values is taken
        dataframe = pandas.DataFrame({'xr': xr_s, 'yr': yr_s, 'values': values_s})
        d = dataframe.groupby(by=['xr', 'yr']).mean().reset_index()
        values = np.array(d['values'])
        xr = np.array(d['xr'])
        yr = np.array(d['yr'])

        projected_matrix = np.nanmean(corr_matrix) * np.ones((np.max(xr) + 1, np.max(yr) + 1))
        projected_matrix[xr, yr] = values
        return projected_matrix, euclidean_distance, angles

    def temporal_reconstruction(self, angle, angles, distances, celerity,
                                correlation_matrix, time_interpolation_resolution):
        D = np.cos(np.radians(angle - angles.T.flatten())) * distances.flatten()
        time = D / celerity
        time_unique, index_unique = np.unique(time, return_index=True)
        index_unique_sorted = np.argsort(time_unique)
        time_unique_sorted = time_unique[index_unique_sorted]
        timevec = np.arange(np.min(time_unique_sorted), np.max(time_unique_sorted),
                            time_interpolation_resolution)
        corr_unique_sorted = correlation_matrix.T.flatten()[index_unique[index_unique_sorted]]
        interpolation = interp1d(time_unique_sorted, corr_unique_sorted)
        SS = interpolation(timevec)
        return SS

    def temporal_reconstruction_tuning(self, SS, time_interpolation_resolution,
                                       low_frequency_ratio, high_frequency_ratio):
        low_frequency = low_frequency_ratio * time_interpolation_resolution
        high_frequency = high_frequency_ratio * time_interpolation_resolution
        sos_filter = scipy.signal.butter(1, (2 * low_frequency, 2 * high_frequency),
                                         btype='bandpass', output='sos')
        SS_filtered = scipy.signal.sosfiltfilt(sos_filter, SS)
        return SS_filtered

    def compute_period(self, SS_filtered, min_peaks_distance):
        peaks_max, properties_max = scipy.signal.find_peaks(SS_filtered, distance=min_peaks_distance)
        period = np.mean(np.diff(peaks_max))
        return period, peaks_max
