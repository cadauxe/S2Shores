#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8 12:01:00 2021

Module containing all depth inversion methods

@author: erwinbergsma
         gregoirethoumyre
         degoulromain
"""

# Imports
import numpy as np
from bathymetry.shoresutils import *

def depth_inversion_with_filter(wave_point,params):

    kKeep=params['NKEEP']
    DIR=wave_point['dir']
    dPHI=wave_point['dPhi']
    T=wave_point['T']
    NU=wave_point['nu']
    CEL=wave_point['cel']
    DCEL=wave_point['dcel']
    DEP = np.empty(kKeep) * np.nan

    for ii in range(0, np.min((DIR.shape[0], kKeep))):
        if (dPHI[ii] != 0) or (np.isnan(dPHI[ii]) == False):
            if (T[ii] > params['MIN_T']) and (T[ii] < params['MAX_T']):
                DEP[ii] = funLinearC_k(NU[ii], CEL[ii],params)
            else:
                NU[ii] = np.nan
                DIR[ii] = np.nan
                CEL[ii] = np.nan
                DCEL[ii] = np.nan
                T[ii] = np.nan
                
    return {
        'depth':DEP,
        'cel': CEL,
        'dcel': DCEL,
        'L': 1 / NU,
        'T': T,
        'dir': DIR
        }