# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8 12:01:00 2021

Module containing all depth inversion methods

@author: erwinbergsma
         gregoirethoumyre
         degoulromain
"""
import numpy as np

from .bathy_physics import funLinearC_k


def depth_linear_inversion(wave_point, config):
    kKeep = config.NKEEP
    DIR = wave_point['dir']
    T = wave_point['T']
    NU = wave_point['nu']
    CEL = wave_point['cel']
    DCEL = wave_point['dcel']
    DEP = np.empty(kKeep) * np.nan

    for ii in range(0, np.min((DIR.shape[0], kKeep))):
        if not np.isnan(CEL[ii]):
            DEP[ii] = funLinearC_k(NU[ii], CEL[ii], config.D_PRECISION, config.G)

    return {
        'depth': DEP,
        'cel': CEL,
        'dcel': DCEL,
        'L': 1 / NU,
        'T': T,
        'dir': DIR
    }
