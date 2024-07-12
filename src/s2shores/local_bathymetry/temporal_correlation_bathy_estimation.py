# -*- coding: utf-8 -*-
""" Class handling the information describing a wave field sample..

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2024 CNES. All rights reserved.
:license: see LICENSE file
:created: 10 September 2021
"""
from ..data_model.bathymetry_sample_estimation import BathymetrySampleEstimation


class TemporalCorrelationBathyEstimation(BathymetrySampleEstimation):
    """ This class encapsulates the information estimated in a bathymetry sample by a
    TemporalCorrelationBathyEstimator.
    """
