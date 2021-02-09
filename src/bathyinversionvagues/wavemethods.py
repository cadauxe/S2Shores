#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 3 10:12:00 2021

Module containing all wave parameters estimation methods

@author: erwinbergsma
         gregoirethoumyre
         degoulromain
"""

# Imports
import numpy as np
import copy
from scipy.signal import find_peaks
from skimage.transform import radon
from bathymetry.shoresutils import *

def spatial_dft_method(Im,params,kfft, phi_min, phi_deep):
    """
    issu de shores.py - funRadonSpecCel_v3_3
    VERSION 3.3 --  QUICKER:    Radon, FFT, find directional peaks, then do 
                                detailed DFT analysis to find detailed phase
                                shifts per linear wave number (k *2pi)
                                
    
    Parameters
    ----------
    Im : numpy.ndarray
        Sub-windowed images in M x N x BANDS -- currently only 2 bands are used!
    kfft : numpy.ndarray
        M x 1 array with LINEAR wave number.
    phi_min : numpy.ndarray
        M x N of size(kfft,theta) that contains the minimum phase shift based on shallow water.
    phi_deep : numpy.ndarray
        M x N of size(kfft,theta) that contains the maximum phase shift based on deep water.
    params : dictionary
        This dictionary contains settings and image specifics.


     Returns
    -------
    dict:
        As output we deliver a dictionary containing 
            -   cel     =   Wave celerity               [m/s]
            -   nu       =   linear Wave number                 [1/m]
            -   L       =   Wavelength                  [m]
            -   T       =   Approximate wave period     [sec]
            -   dir     =   Wave direction (RADON)      [degrees]
            -   dPhi    =   Measured phase shift        [rad]

    """
    
    # pre-set all output variables:)
    kKeep=params['NKEEP']
    CEL = np.empty(kKeep) * np.nan  # Estimated celerity
    DIR = np.empty(kKeep) * np.nan
    dPHI = np.empty(kKeep) * np.nan
    PHIRat = np.empty(kKeep) * np.nan
    NU = np.empty(kKeep) * np.nan
    Emax = np.empty(kKeep) * np.nan
    T = np.empty(kKeep) * np.nan
    DCEL = np.empty(kKeep) * np.nan
    
    # All rotational angles (theta) for the Radon Transform
    thetaFFT = np.linspace(params['MIN_DIR'], params['MAX_DIR'], params['MAX_NDIRS'], endpoint=False)
  
   
   
    # Check if the image is NOT empty (if statement):
    if sc_all(Im):
        # Create Radon sinograms per sub image [we can make this a for loop for the number of frames]
        sinogram1 = radon(funDetrend_2d(Im[:, :, 0]), theta=thetaFFT)
        sinogram2 = radon(funDetrend_2d(Im[:, :, 1]), theta=thetaFFT)
        # signal length to normalise the spectrum:
        N = sinogram1.shape[0]
        # Retrieve total spectrum, controlled by physical wave propagatinal limits:
        totalSpecFFT, _, _, phase_check = funGetSpectralPeaks(Im, thetaFFT, params)
        # Find maximum total energy per direction theta
        totalSpecMaxheta = np.max(totalSpecFFT, axis=0) / np.max(np.max(totalSpecFFT, axis=0))
        # Pick the maxima 
        peaksDir = find_peaks(totalSpecMaxheta, prominence=params['PROMINENCE_MAX_PEAK'])
        
        if peaksDir[0].size > 0:
            for ii in range(0, peaksDir[0].size):
                tmp = np.arange(np.max([peaksDir[0][ii] - params['ANGLE_AROUND_PEAK_DIR'], 0]), np.min([peaksDir[0][ii] + params['ANGLE_AROUND_PEAK_DIR']+1, 360]))
                if ii == 0:
                    dirInd = tmp
                else:
                    dirInd = np.append(dirInd, tmp)

            # delete double directions:        
            dirInd = np.unique(dirInd)
            # create phi limits matrix
            phi_min = np.tile(phi_min[:, np.newaxis], (1, dirInd.shape[0]))
            phi_deep = np.tile(phi_deep[:, np.newaxis], (1, dirInd.shape[0]))
            thetaTmp = thetaFFT[dirInd]
            # Detailed analysis of the signal for positive phase shifts
            sinoFFT1 = np.empty((kfft.size, dirInd.shape[0])) * (np.nan + 0.j)
            sinoFFT2 = np.empty((kfft.size, dirInd.shape[0])) * (np.nan + 0.j)

            for ii in range(0, dirInd.shape[0]):
                sinoFFT1[:, ii] = DFT_fr(sinogram1[:, dirInd[ii]], kfft, 1 / params['DX'])
                sinoFFT2[:, ii] = DFT_fr(sinogram2[:, dirInd[ii]], kfft, 1 / params['DX'])

            sinoFFt = np.dstack((sinoFFT1, sinoFFT2))
            # This allows to calucalate the phase, amplitude:
            phase_shift = np.angle(sinoFFt[:, :, 1] * np.conj(sinoFFt[:, :, 0]))
            # the phase comes between -pi and pi but we want to know the fraction of the total wave thus  0 < dphi < 2pi
            phase_shift_unw = copy.deepcopy(phase_shift)

            if params['UNWRAP_PHASE_SHIFT'] == False:
                # currently deactivated but we want this functionality:
                phase_shift_unw = np.abs(phase_shift_unw)
            else:
                phase_shift_unw = (phase_shift_unw + 2 * np.pi) % (2 * np.pi)

            # Deep water limitation [if the wave travels faster that the deep-water limit we consider it non-physical]
            phase_shift[phase_shift_unw > phi_deep[:, :dirInd.shape[0]]] = 0
            phase_shift_unw[phase_shift_unw > phi_deep[:, :dirInd.shape[0]]] = 0
            # Minimal propagation speed; this depends on the Satellite; Venus or Sentinel 2
            phase_shift[phase_shift_unw < phi_min[:, :dirInd.shape[0]]] = 0
            phase_shift_unw[phase_shift_unw < phi_min[:, :dirInd.shape[0]]] = 0
            totSpec = (np.abs(((np.abs(sinoFFt[:, :, 0]) ** 2 + np.abs(sinoFFt[:, :, 1]) ** 2) / (N ** 2)) * phase_shift_unw) / N)
            # Refined spectral solution:
            totalSpecMax_ref = np.max(totSpec, axis=0) / np.max(np.max(totSpec, axis=0))
            peaksDir = find_peaks(totalSpecMax_ref, prominence=params['PROMINENCE_MULTIPLE_PEAKS'])
            peaksDir = peaksDir[0][:kKeep]

            if peaksDir.size > 0:
                Emax[:len(peaksDir)] = totalSpecMax_ref[peaksDir]
                DIR[:len(peaksDir)] = thetaTmp[peaksDir]
                peaksK = np.argmax(totSpec[:, peaksDir], axis=0)
                NU[:len(peaksDir)] = kfft[peaksK].squeeze()
                dPHI[:len(peaksDir)] = phase_shift_unw[peaksK, peaksDir]
                PHIRat[:len(peaksDir)] = dPHI[:len(peaksDir)] / phi_deep[peaksK, peaksDir]
                CEL = dPHI / (2 * np.pi * NU * params['DT'])
                T = 1 / (CEL * NU)
                 
                # sort now on longest waves:
                sorting = np.argsort(-((PHIRat ** 2) * Emax))

                CEL = CEL[sorting]
                DCEL = DCEL[sorting]
                NU = NU[sorting]
                T = T[sorting]
                DIR = DIR[sorting]
                dPHI = dPHI[sorting]


    return {'cel': CEL,
            'nu': NU,
            'L': 1 / NU,
            'T': T,
            'dir': DIR,
            'dPhi': dPHI,
            'dcel': DCEL
            }
