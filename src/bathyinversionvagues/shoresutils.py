#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 3 10:12:00 2021

Module containing all functions common to waves and bathy estimation methods

@author: erwinbergsma
         gregoirethoumyre
         degoulromain

"""
import numpy as np
from skimage.transform import radon
from scipy.signal import convolve2d
from scipy.signal import detrend
from scipy.signal import medfilt2d
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
import pandas
import scipy

# check array is not empty
def sc_all(array):
    for x in array.flat:
        if not x:
            return False
    return True


def funDetrend_2d(Z):
    M = Z.shape[1]
    N = Z.shape[0]

    XX, YY = np.meshgrid(range(0, M), range(0, N))
    Xcolv = XX.flatten()
    Ycolv = YY.flatten()
    Zcolv = Z.flatten()

    Mp = np.zeros((len(Ycolv), 3))
    Const = np.ones(len(Xcolv))
    Mp[:, 0] = Xcolv
    Mp[:, 1] = Ycolv
    Mp[:, 2] = Const

    inv_Mp = np.linalg.pinv(Mp)

    Coef = np.dot(inv_Mp, Zcolv)
    Coef = np.asarray(Coef)

    XCoeff = Coef[0]
    YCoeff = Coef[1]
    CCoeff = Coef[2]

    Z_p = XCoeff * XX + YCoeff * YY + CCoeff
    Z_f = Z - Z_p

    return (Z_f)

def funSinoFFT(sino1, sino2, dx):
    '''
    Parameters
    ----------
    I : 2D np.array that contains NxMx2 structure.
        Typically these are two different images at the same location at a different moment in time.
    dx : Spatial resolution of the image I.
        DESCRIPTION. The default is 10 (sentinel)

    Returns
    -------
    out : Dictioinary with all infomation on the Radon spectrum
        DESCRIPTION.

    '''
    Nx = np.shape(sino1)[0]
    Fs = 1 / float(dx)
    kfft = np.arange(0, Fs / 2, Fs / Nx) + (Fs / (2 * Nx))

    sino1_fft = []
    sino2_fft = []

    for ii in range(0, sino1.shape[1]):
        Y1 = np.fft.fft(sino1[:, ii])
        Y2 = np.fft.fft(sino2[:, ii])

        if (len(Y1) > 1):
            Yout1 = Y1[0:int(np.ceil(Nx / 2))]
            Yout2 = Y2[0:int(np.ceil(Nx / 2))]
        else:
            Yout1 = Y1
            Yout2 = Y2

        sino1_fft.append(Yout1)
        sino2_fft.append(Yout2)

    sinoFFT = np.dstack((np.array(sino1_fft).transpose(), \
                         np.array(sino2_fft).transpose()))

    return sinoFFT, kfft, Nx

def funGetSpectralPeaks(Im, theta, unwrap_phase_shift,dt,dx,min_D,g):
    sinogram1 = radon(funDetrend_2d(Im[:, :, 0]), theta=theta)
    sinogram2 = radon(funDetrend_2d(Im[:, :, 1]), theta=theta)

    sinoFFT, kfft, N = funSinoFFT(sinogram1, sinogram2, dx)

    combAmpl = (np.abs(sinoFFT[:, :, 0]) ** 2 + np.abs(sinoFFT[:, :, 1]) ** 2) / (N ** 2)

    phase_shift = np.angle(sinoFFT[:, :, 1] * np.conj(sinoFFT[:, :, 0]))

    if unwrap_phase_shift == False:
        # currently deactivated but we want this functionality:
        phase_shift = phase_shift
    else:
        phase_shift = (phase_shift + 2 * np.pi) % (2 * np.pi)

    # deep water limits:
    phi_deep = (2 * np.pi * dt) / (np.sqrt(1 / (np.round(g/(2*np.pi),2)* kfft))).squeeze()
    phi_deep = np.tile(phi_deep[:, np.newaxis], (1, theta.shape[0]))

    # shallow water limits:
    min_cel = np.sqrt(g * min_D)
    phi_min = (2 * np.pi * dt * kfft * min_cel).squeeze()
 
    phi_min = np.tile(phi_min[:, np.newaxis], (1, theta.shape[0]))

    # Deep water limitation [if the wave travels faster that the deep-water limit we consider it non-physical]
    phase_shift[np.abs(phase_shift) > phi_deep] = 0

    # Minimal propagation speed; this depends on the Satellite; Venus or Sentinel 2
    phase_shift[np.abs(phase_shift) < phi_min] = 0

    return (combAmpl * phase_shift) / N, kfft, N, phase_shift


def DFT_fr(x, fr, fs):
    """
    Compute the discrete Fourier Transform of the 1D array x
    :param x: (array)
    """

    N = x.size
    n = np.arange(N)
    # k = n.reshape((N, 1))

    e = np.exp(-2j * np.pi * fr * n / fs)
    return np.dot(e, x)

def funConv2(x, y, mode='same'):
    '''


    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    mode : TYPE, optional
        DESCRIPTION. The default is 'same'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

def funSmoothc(mI, Nr, Nc):
    '''

    Parameters
    ----------
    mI : TYPE
        DESCRIPTION.
    Nr : TYPE
        DESCRIPTION.
    Nc : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    '''
    % SMOOTHC.M: Smooths matrix data, cosine taper.
    % MO=SMOOTHC(MI,Nr,Nc) smooths the data in MI
    % using a cosine taper over 2*N+1 successive points, Nr, Nc points on
    % each side of the current point.
    %
    % Inputs: mI - original matrix
    % Nr - number of points used to smooth rows
    % Nc - number of points to smooth columns
    % Outputs:mO - smoothed version of original matrix
    %
    %
    '''

    # Determine convolution kernel k
    Nr = Nr + 1
    Nc = Nc + 1

    kr = 2 * Nr + 1
    kc = 2 * Nc + 1

    midr = Nr + 1
    midc = Nc + 1
    maxD = (Nr ** 2 + Nc ** 2) ** 0.5

    k = np.zeros((kr, kc))
    for irow in range(0, kr):
        for icol in range(0, kc):
            D = np.sqrt(((midr - irow) ** 2) + ((midc - icol) ** 2))
            k[irow, icol] = np.cos(D * np.pi / 2 / maxD);

    k = k / np.sum(k.ravel())
    # Perform convolution
    out = funConv2(mI, k, 'same')
    return out[Nr:-Nr, Nc:-Nc]


def funSmooth2(M, nx, ny):
    '''
    Parameters
    ----------
    M : TYPE
        DESCRIPTION.
    nx : TYPE
        DESCRIPTION.
    ny : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    S = np.concatenate((np.tile(M[0, :], (nx, 1)).transpose(), \
                        M.transpose(), \
                        np.tile(M[-1, :], (nx, 1)).transpose()), axis=1).transpose()

    S = np.concatenate((np.tile(S[:, 0], (ny, 1)).transpose(), \
                        S, \
                        np.tile(S[:, -1], (ny, 1)).transpose()), axis=1)

    S = funSmoothc(S, nx - 1, ny - 1)

    return (S)

def funLinearC_k(nu, c,d_precision,d_init,g):
    k = 2 * np.pi * nu  #angular wave number
    precision = d_precision
    w = c * k
    do = d_init
    d = c ** 2 / g

    while (abs(do - d) > precision):
        do = d
        dispe = w ** 2 - (g * k * np.tanh(k * d))
        fdispe = -g * (k ** 2) / (np.cosh(k * d) ** 2)
        d = d - (dispe / fdispe)

    return (d)

def fft_filtering(simg,spatial_resolution,T_max,T_min):
    """
    Compute the fft filtering of a subtile
    :param simg:(np.array) the given sequence of images to filter
    :param spatial_resolution: (int) sampling resolution (default 10 meters on Sentinel 2)
    :param T_max:(int) Max wave periode
    :param T_min:(int) Min wave periode
    :return: simg_filtered:
    """
    flag = 0
    n,m,c = simg.shape
    kx = np.fft.fftshift(np.fft.fftfreq(n, spatial_resolution))
    ky = np.fft.fftshift(np.fft.fftfreq(m, spatial_resolution))
    kx = np.repeat(np.reshape(kx, (n, 1)), m, axis=1)
    ky = np.repeat(np.reshape(ky, (1, m)), n, axis=0)
    threshold_min = 1 / (1.56 * T_max ** 2)
    threshold_max = 1 / (1.56 * T_min ** 2)
    simg_filtered = np.zeros(simg.shape)
    kr = np.sqrt(kx ** 2 + ky ** 2)
    kr[kr < threshold_min] = 0
    kr[kr > threshold_max] = 0
    boolKr = (kr > 0)
    for channel in range(c):
        r = simg[:, :,channel]
        r = detrend(detrend(r, axis=1), axis=0)
        fftr = np.fft.fft2(r)
        energy_r = np.fft.fftshift(fftr)
        energy_r *= boolKr
        max_energy = np.max(np.abs(energy_r))
        if max_energy > 3 or max_energy < 0.01:
            flag = 1
            simg_filtered[:, :,channel] = np.real(np.fft.ifft2(np.fft.ifftshift(energy_r)))
    return simg_filtered, flag

def cross_correlation(A, B):
    """
        Compute the correlation of each line of A with each line of B
        This function is faster than using np.corrcoef which computes correlation beetween A&A,A&B,B&A,B&B
        :param A (np.array) : matrix A
        :param B (np.array) : matrix B
        :return: sub_tile (np.array) : cross correlation matrix
        """
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))

def filter_mean(time_serie,window):
    if len(time_serie)<2*window :
        raise ValueError("time serie is too small compared to the window")
    else:
        padded_time_serie = np.concatenate((np.full(window,np.mean(time_serie[:window])),time_serie,np.full(window,np.mean(time_serie[-(window+1):]))),axis=0)
        return np.convolve(padded_time_serie,np.ones(2*window+1)/(2*window+1),'valid')

def permute_axes(Im):
    n1,n2,n3 = np.shape(Im)
    pIm = np.zeros((n2,n3,n1))
    for i in np.arange(n1):
        pIm[:,:,i] = Im[i,:,:]
    return pIm

def create_sequence_time_series_temporal(Im,percentage_points,spatial_resolution,fft_T_max,fft_T_min):
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
            simg_filtered (numpy array of size (number_of_random_points,number_of_frames)) : sequence of filtered (pass band) thumbnails for debug purposes
    """
    nx, ny ,nframes= np.shape(Im)
    simg_filtered, flag = fft_filtering(Im,spatial_resolution=spatial_resolution,T_max=fft_T_max,T_min=fft_T_min)
    array = np.reshape(simg_filtered, (nx*ny,-1))
    nb_random_points = round(nx*ny*percentage_points/100)
    random_indexes = np.random.randint(0,nx*ny,size=nb_random_points)
    yy,xx = np.meshgrid(np.linspace(1,nx,nx),np.linspace(1,ny,ny))
    xx = xx.flatten()
    yy = yy.flatten()
    return (array[random_indexes,:],xx[random_indexes],yy[random_indexes],simg_filtered)

def create_sequence_time_series_spatial(Im,spatial_resolution,fft_T_max,fft_T_min):
    """
    This function takes a sequence of images and filters (passband) sequence and takes random time series within the thumbnail.
    Random are not used and array is not flat
    :param Im (numpy array of size (number_of_lines,number_of_column,number_of_frames)) : sequence of thumbnails
    :param spatial_resolution (int) : percentage of points taken within the thumbnail
    :param fft_T_max (int) : Max wave period to be allowed
    :param fft_T_min (int) : Min wave period to be allowed
    :return sequence_images (numpy array of size (number_of_random_points,number_of_frames)) : sequence of filtered (pass band) thumbnails
            xx (flatten numpy array of size number_of_random_points) : list x value of random points
            yy (flatten numpy array of size number_of_random_points) : list y value of random points
    """
    nx, ny, nframes = np.shape(Im)
    array, flag = fft_filtering(Im, spatial_resolution=spatial_resolution,T_max=fft_T_max,T_min=fft_T_min)
    xx = np.arange(nx)
    yy = np.arange(ny)
    return (array, xx, yy)

def compute_angles_distances(M):
    (n1,n2,n3) = np.shape(M)
    xx = np.array([np.arange(n1)])
    yy = np.array([np.arange(n2)])
    dxx = xx - xx.T
    dyy = yy - yy.T
    distances = np.sqrt(np.square((dxx)) + np.square((dyy)))
    angles = np.angle(dxx+1j*dyy)
    return (angles,distances)

def compute_temporal_correlation(sequence_thumbnail,number_frame_shift):
    """
        This function computes the correlation of each time serie of sequence_thumbnail with each time serie of sequence_thumbnail but shifted of number_frame_shift frames
        :param sequence_thumbnail (numpy array of size (number_of_frames,number_of_time_series)) : video of waves
        :param number_frame_shift (int) : number of shifted frames
        :return corr (numpy array of size (number_of_time_series,number_of_time_series)) : cross correlation of time series
        """
    corr = cross_correlation(sequence_thumbnail[:,number_frame_shift:], sequence_thumbnail[:,:-number_frame_shift])
    return corr


def normxcorr2(template, image, mode="full"):
    ########################################################################################
    # Author: Ujash Joshi, University of Toronto, 2017                                     #
    # Based on Octave implementation by: Benjamin Eltzner, 2014 <b.eltzner@gmx.de>         #
    # Octave/Matlab normxcorr2 implementation in python 3.5                                #
    # Details:                                                                             #
    # Normalized cross-correlation. Similiar results upto 3 significant digits.            #
    # https://github.com/Sabrewarrior/normxcorr2-python/master/norxcorr2.py                #
    # http://lordsabre.blogspot.ca/2017/09/matlab-normxcorr2-implemented-in-python.html    #
    ########################################################################################
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the �full� output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """
    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
                    len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return out

def compute_spatial_correlation(sequence_thumbnail,number_frame_shift):
    size_x, size_y, number_frames = np.shape(sequence_thumbnail)
    full_corr = normxcorr2(sequence_thumbnail[:,:,0].T, sequence_thumbnail[:,:,number_frame_shift].T)
    for index in np.arange(number_frame_shift,number_frames-number_frame_shift,number_frame_shift):
        corr = normxcorr2(sequence_thumbnail[:,:,index].T, sequence_thumbnail[:,:,index+number_frame_shift].T)
        full_corr = full_corr + corr
    return full_corr

def cartesian_projection(corr_matrix,xx,yy,spatial_resolution):
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

    dataframe = pandas.DataFrame({'xr': xr_s, 'yr': yr_s, 'values': values_s})
    d = dataframe.groupby(by=['xr', 'yr']).mean().reset_index()
    values = np.array(d['values'])
    xr = np.array(d['xr'])
    yr = np.array(d['yr'])

    projected_matrix = np.nanmean(corr_matrix) * np.ones((np.max(xr) + 1, np.max(yr) + 1))
    projected_matrix[xr, yr] = values
    return (projected_matrix, euclidean_distance, angles)

def correlation_tuning(correlation_matrix,ratio):
    """
        This function tunes carrelation matrix by detrending the signal and removing the edges.
        :param correlation_matrix (numpy array) : entry correlation matrix
        :param ratio (float) : edges to be remove (1 takes all signal and 0 remove all signal)
        :return correlation_matrix (numpy array) : correlation matrix tuned
        """
    correlation_matrix[np.isnan(correlation_matrix)]=np.nanmedian(correlation_matrix)
    correlation_matrix = funDetrend_2d(correlation_matrix)
    s1, s2 = np.shape(correlation_matrix)
    corr_car_tuned = correlation_matrix[int(s1 / 2-ratio * s1 /2):int(s1 / 2+ratio * s1 /2),int(s2 / 2-ratio * s2 /2):int(s2 / 2+ratio * s2 /2)]
    return corr_car_tuned

def compute_sinogram(correlation_matrix,median_filter_kernel_ratio,mean_filter_kernel_size):
    """
    This function take a correlation matrix, compute the radon transform of this matrix and return the sinogram for the angle which maximizes the variance of the radon transform.
    The purpose here is to find direction.
    :param correlation_matrix (numpy array of size (number_of_lines,number_of_column)) : correlation matrix
           median_filter_kernel_ratio (float) : median filter applied on radon matrix angles. 1 uses a window of the signal length
           mean_filter_kernel_size (int) : mean filter window size applied on variance
    :return: sinogram_max_var (numpy array in one dimension) : sinogram maximizing the variance
             argmax_var (float) : angle maximizing the variance
    """
    theta = np.arange(0,180)
    radon_matrix = radon(correlation_matrix, theta=theta, circle=True)
    kernel_size_1 = round(median_filter_kernel_ratio*np.shape(radon_matrix)[0])
    if (kernel_size_1%2)==0:
        kernel_size_1 = kernel_size_1 +1
    kernel_size_2 = 3
    # each element of kernel must be odd
    radon_matrix_tuned = radon_matrix - medfilt2d(radon_matrix,kernel_size=(kernel_size_1,kernel_size_2))
    variance = filter_mean(np.var(radon_matrix_tuned,axis=0),mean_filter_kernel_size)
    propagation_angle = np.argmax(variance)
    sinogram_max_var = radon_matrix[:,propagation_angle]
    return (sinogram_max_var,propagation_angle,variance,radon_matrix)

def sinogram_tuning(sinogram,mean_filter_kernel_size):
    """
    This function tuned the sinogram using mean filter
    :param sinogram (numpy array) : sinogram
           mean_filter_kernel_size (int) : mean filter window size applied on sinogram
    :return: sinogram_max_var_tuned (numpy array) : tuned sinogram
    """
    sinogram_max_var_tuned=filter_mean(sinogram,mean_filter_kernel_size)
    return sinogram_max_var_tuned

def compute_wave_length(sinogram):
    """
    This function computes wave length of the signal using the zeros of the signal
    :param sinogram (numpy array) : sinogram
    :return: wave_length (numpy array) : wave length
             zeros (numpy array) : zeros of the signal
    """
    sign = np.sign(sinogram)
    diff = np.diff(sign)
    zeros = np.where(diff!=0)[0]
    wave_length = 2 * np.mean(np.diff(zeros))
    return (wave_length,zeros)

def compute_celerity(sinogram,wave_length,spatial_resolution,time_resolution,temporal_lag):
    """
    This function computes celerity of waves
    :param sinogram (numpy array) : sinogram
           wave_length (float) : wave length in meter
           time_resolution (float) : time resolution in second
           temporal_lag (float) : temporal lag in frames
    :return: celerity (numpy array) : wave length
             argmax (int) : position of maximum used to compute celerity
    """
    size_sinogram = len(sinogram)
    m1 = max(int(size_sinogram/2-wave_length/2),0)
    m2 = min(int(size_sinogram/2+wave_length/2),size_sinogram)
    argmax = np.argmax(sinogram[m1:m2])
    rhomx = spatial_resolution * np.abs(argmax +m1 - size_sinogram/2)
    t = time_resolution * temporal_lag
    celerity = np.abs(rhomx / t)
    return (celerity, argmax+m1)

def temporal_reconstruction(angle,angles,distances,celerity,correlation_matrix,time_interpolation_resolution):
    D = np.cos(np.radians(angle-angles.T.flatten()))*distances.flatten()
    time=D/celerity
    time_unique,index_unique = np.unique(time,return_index=True)
    index_unique_sorted = np.argsort(time_unique)
    time_unique_sorted = time_unique[index_unique_sorted]
    timevec = np.arange(np.min(time_unique_sorted), np.max(time_unique_sorted), time_interpolation_resolution)
    corr_unique_sorted = correlation_matrix.T.flatten()[index_unique[index_unique_sorted]]
    interpolation = interp1d(time_unique_sorted,corr_unique_sorted)
    SS = interpolation(timevec)
    return SS

def temporal_reconstruction_tuning(SS,time_interpolation_resolution,low_frequency_ratio,high_frequency_ratio):
    low_frequency = low_frequency_ratio * time_interpolation_resolution
    high_frequency = high_frequency_ratio * time_interpolation_resolution
    sos_filter = scipy.signal.butter(1, (2*low_frequency, 2*high_frequency), btype='bandpass', output='sos')
    SS_filtered = scipy.signal.sosfiltfilt(sos_filter, SS)
    return SS_filtered

def compute_period(SS_filtered,min_peaks_distance):
    peaks_max, properties_max = scipy.signal.find_peaks(SS_filtered, distance=min_peaks_distance)
    period = np.mean(np.diff(peaks_max))
    return (period,peaks_max)