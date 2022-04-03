# -*- coding: utf-8 -*-
""" Class handling the information describing a waves field sample..

:author: Grégoire Thoumyre
:organization: CNES/LEGOS
:copyright: 2021 CNES/LEGOS. All rights reserved.
:license: see LICENSE file
:created: 20 sep 2021
"""
from ..data_model.bathymetry_sample_estimation import BathymetrySampleEstimation


class SpatialCorrelationWavesFieldEstimation(BathymetrySampleEstimation):
    """ This class encapsulates the information estimated in a waves field sample by a
    SpatialCorrelationBathyEstimator.

    It defines the estimation attributes specific to this estimator.
    """
