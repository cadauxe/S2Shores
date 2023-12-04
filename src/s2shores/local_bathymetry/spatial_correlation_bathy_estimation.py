# -*- coding: utf-8 -*-
""" Class handling the information describing a wave field sample..

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2022 CNES. All rights reserved.
:license: see LICENSE file
:created: 29 novembre 2022
"""
from ..data_model.bathymetry_sample_estimation import BathymetrySampleEstimation


class SpatialCorrelationBathyEstimation(BathymetrySampleEstimation):
    """ This class encapsulates the information estimated in a wave field sample by a
    SpatialCorrelationBathyEstimator.

    It defines the estimation attributes specific to this estimator.
    """
