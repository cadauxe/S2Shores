# -*- coding: utf-8 -*-
""" Class handling the information describing a wave field sample..

:author: Grégoire Thoumyre
:organization: CNES/LEGOS
:copyright: 2021 CNES/LEGOS. All rights reserved.
:license: see LICENSE file
:created: 20 sep 2021
"""
from ..data_model.bathymetry_sample_estimation import BathymetrySampleEstimation


class SpatialCorrelationBathyEstimation(BathymetrySampleEstimation):
    """ This class encapsulates the information estimated in a wave field sample by a
    SpatialCorrelationBathyEstimator.

    It defines the estimation attributes specific to this estimator.
    """
