# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:45:01 2020

This is a bathymetry inversion package with all kinds of functions for
depth inversion. Initially designed for TODO

@author: erwinbergsma
         gregoirethoumyre
"""
import numpy as np


def sc_all(array):
    for x in array.flat:
        if not x:
            return False
    return True


def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res


def permute_axes(Im):
    n1, n2, n3 = np.shape(Im)
    pIm = np.zeros((n2, n3, n1))
    for i in np.arange(n1):
        pIm[:, :, i] = Im[i, :, :]
    return pIm
