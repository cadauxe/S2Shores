# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8 12:01:00 2021

Module containing all depth inversion methods

@author: erwinbergsma
         gregoirethoumyre
         degoulromain
"""
import copy

import numpy as np

from .bathy_physics import funLinearC_k


# TODO: remove this function and rely on waves_field_samples attributes
def depth_linear_inversion(wave_point, config):
    kKeep = config.NKEEP
    NU = wave_point['nu']
    CEL = wave_point['cel']
    DEP = np.empty(kKeep) * np.nan

    for ii in range(0, np.min((CEL.shape[0], kKeep))):
        if not np.isnan(CEL[ii]):
            DEP[ii] = funLinearC_k(NU[ii], CEL[ii], config.D_PRECISION, config.G)

    wave_point_out_dic = copy.deepcopy(wave_point)
    wave_point_out_dic['depth'] = DEP
    wave_point_out_dic['L'] = 1 / NU
    return wave_point_out_dic
