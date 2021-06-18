# -*- coding: utf-8 -*-
"""
Created on Wed Feb 3 10:12:00 2021

Module containing all wave parameters estimation methods

@author: erwinbergsma
         gregoirethoumyre
         degoulromain
"""
import numpy as np

from .waves_field_estimation import WavesFieldEstimation


def build_correlation_output(direction, wavelength, period, celerity, config):
    # FIXME: DT does not seem to be the right value to take here. Use DeltaTimeProvider when written
    waves_field_estimation = WavesFieldEstimation(config.DT,
                                                  config.D_PRECISION,
                                                  config.G,
                                                  config.DEPTH_EST_METHOD)
    waves_field_estimation.direction = direction
    waves_field_estimation.wavelength = wavelength
    waves_field_estimation.period = period
    waves_field_estimation.celerity = celerity

    waves_fieldestimation_as_dict = {'cel': np.array([waves_field_estimation.celerity]),
                                     'nu': np.array([waves_field_estimation.wavenumber]),
                                     'L': np.array([waves_field_estimation.wavelength]),
                                     'T': np.array([waves_field_estimation.period]),
                                     'dir': np.array([waves_field_estimation.direction]),
                                     'depth': np.array([waves_field_estimation.depth]),
                                     'dcel': np.array([0])
                                     }

    return waves_field_estimation, waves_fieldestimation_as_dict
