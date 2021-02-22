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
from bathycommun.src.bathycommun.config.config_bathy import ConfigBathy
import numpy as np
import os
from pathlib import Path
import copy
from scipy.signal import find_peaks
from skimage.transform import radon
from shoresutils import *

yaml_file = 'config/wave_bathy_inversion_config.yaml'
config = ConfigBathy(os.path.join(Path(os.path.dirname(__file__)).parents[1],yaml_file))

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
        totalSpecFFT, _, _, phase_check = funGetSpectralPeaks(Im, thetaFFT, params['UNWRAP_PHASE_SHIFT'],params['DT'],params['DX'],params['MIN_D'],params['G'])
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

                for ii in range(0, np.min((DIR.shape[0], kKeep))):
                    if (dPHI[ii] != 0) or (np.isnan(dPHI[ii]) == False):
                         if (T[ii] <= params['MIN_T']) or (T[ii] >= params['MAX_T']):
                            NU[ii] = np.nan
                            DIR[ii] = np.nan
                            CEL[ii] = np.nan
                            DCEL[ii] = np.nan
                            T[ii] = np.nan
                            PHIRat[ii] = np.nan
                
                # sort now on longest waves:
                sorting = np.argsort(-((PHIRat ** 2) * Emax))

                CEL = CEL[sorting]
                DCEL = DCEL[sorting]
                NU = NU[sorting]
                T = T[sorting]
                DIR = DIR[sorting]

    return {'cel': CEL,
            'nu': NU,
            'T': T,
            'dir': DIR,
            'dcel': DCEL
            }

def temporal_correlation_method(Im):
    """
    Bathymetry computation function based on time series correlation

    Parameters
    ----------
    Im : numpy.ndarray
        Sub-windowed images in M x N x BANDS
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

    """
    try:
        stime_series, xx , yy , simg_filtered= create_sequence_time_series_temporal(Im=Im,spatial_resolution=config.temporal_method.resolution.spatial,percentage_points=config.temporal_method.percentage_points,fft_T_max=config.preprocessing.passband.high_period,fft_T_min=config.preprocessing.passband.low_period)
        corr = compute_temporal_correlation(sequence_thumbnail=stime_series, number_frame_shift=config.temporal_method.temporal_lag)
        corr_car, distances, angles = cartesian_projection(corr_matrix=corr, xx=xx, yy=yy, spatial_resolution=config.temporal_method.resolution.spatial)
        corr_car_tuned = correlation_tuning(correlation_matrix=corr_car, ratio=config.temporal_method.tuning.ratio_size_correlation)
        (sinogram_max_var, angle, variance, radon_matrix) = compute_sinogram(correlation_matrix=corr_car_tuned,
                                                                                       median_filter_kernel_ratio=config.temporal_method.tuning.median_filter_kernel_ratio_sinogram,
                                                                                       mean_filter_kernel_size=config.temporal_method.tuning.mean_filter_kernel_size_sinogram)
        sinogram_tuned = sinogram_tuning(sinogram=sinogram_max_var, mean_filter_kernel_size=config.temporal_method.tuning.mean_filter_kernel_size_sinogram)
        wave_length, zeros = compute_wave_length(sinogram=sinogram_tuned)
        celerity, argmax = compute_celerity(sinogram=sinogram_tuned, wave_length=wave_length,spatial_resolution=config.temporal_method.resolution.spatial,
                                                         time_resolution=config.temporal_method.resolution.temporal,
                                                         temporal_lag=config.temporal_method.temporal_lag)
        SS = temporal_reconstruction(angle=angle, angles=np.degrees(angles), distances=distances, celerity=celerity, correlation_matrix=corr,
                                               time_interpolation_resolution=config.temporal_method.resolution.time_interpolation)
        SS_filtered = temporal_reconstruction_tuning(SS, time_interpolation_resolution=config.temporal_method.resolution.time_interpolation,
                                                               low_frequency_ratio=config.temporal_method.tuning.low_frequency_ratio_temporal_reconstruction, high_frequency_ratio=config.temporal_method.tuning.high_frequency_ratio_temporal_reconstruction)
        T, peaks_max = compute_period(SS_filtered=SS_filtered, min_peaks_distance=config.temporal_method.tuning.min_peaks_distance_period)
        return {'cel': celerity,
                'nu': 1 / wave_length,
                'T': T,
                'dir': angle,
                'dcel': 0
                }
    except:
        print("Bathymetry computation failed")