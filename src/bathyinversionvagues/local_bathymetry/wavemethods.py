# -*- coding: utf-8 -*-
"""
Created on Wed Feb 3 10:12:00 2021

Module containing all wave parameters estimation methods

@author: erwinbergsma
         gregoirethoumyre
         degoulromain
"""
from .waves_field_estimation import WavesFieldEstimation


# TODO: generalize as a  local_bathy_estimator method taking into account the anomaly l=c*t
def build_waves_field_estimation(direction: float, wavelength: float, period: float,
                                 celerity: float, config) -> WavesFieldEstimation:
    # FIXME: DT does not seem to be the right value to take here. Use DeltaTimeProvider when written
    waves_field_estimation = WavesFieldEstimation(config.DT,
                                                  config.D_PRECISION,
                                                  config.G,
                                                  config.DEPTH_EST_METHOD)
    waves_field_estimation.direction = direction
    waves_field_estimation.wavelength = wavelength
    waves_field_estimation.period = period
    waves_field_estimation.celerity = celerity

    return waves_field_estimation
