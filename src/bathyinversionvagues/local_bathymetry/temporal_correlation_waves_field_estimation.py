# -*- coding: utf-8 -*-
""" Class handling the information describing a waves field sample..

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 10 sep 2021
"""
from ..data_model.waves_field_estimation import WavesFieldEstimation


class TemporalCorrelationWavesFieldEstimation(WavesFieldEstimation):
    """ This class encapsulates the information estimated in a waves field sample by a
    TemporalCorrelationBathyEstimator.

    At the moment there is no estimation attributes specific to this estimator.
    """
