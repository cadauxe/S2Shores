# -*- coding: utf-8 -*-
""" Class performing bathymetry computation using temporal correlation method

:author: Degoul Romain
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 18/06/2021
"""
from ..local_bathymetry.local_bathy_estimator import LocalBathyEstimatorDebug
from ..local_bathymetry.temporal_correlation_bathy_estimator import \
    TemporalCorrelationBathyEstimator

from .debug_display import temporal_method_debug


class TemporalCorrelationBathyEstimatorDebug(LocalBathyEstimatorDebug,
                                             TemporalCorrelationBathyEstimator):
    """ Class performing debugging for temporal correlation method
    """

    def explore_results(self) -> None:
        temporal_method_debug(self)
