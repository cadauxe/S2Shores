# -*- coding: utf-8 -*-
""" Selection of the desired local bathymetry estimator

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 18/06/2021
"""
from typing import Type  # @NoMove

from .spatial_correlation_bathy_estimator import SpatialCorrelationBathyEstimator
from .spatial_dft_bathy_estimator import SpatialDFTBathyEstimator
from .temporal_correlation_bathy_estimator import TemporalCorrelationBathyEstimator


# Dictionary of classes to be instanciated for each local bathymetry estimator
LOCAL_BATHY_ESTIMATION_CLS = {'SPATIAL_DFT': SpatialDFTBathyEstimator,
                              'TEMPORAL_CORRELATION': TemporalCorrelationBathyEstimator,
                              'SPATIAL_CORRELATION': SpatialCorrelationBathyEstimator}


def get_local_bathy_estimator(local_estimator_code: str) -> Type:
    """ return the local bathymetry estimator class corresponding to a given estimator code

    :param local_estimator_code: the code of the bathymetry estimator class
    :returns: the function to call for computing local bathymetry
    :raises NotImplementedError: when the requested bathymetry estimator is unknown
    """
    try:
        local_bathy_estimator_cls = LOCAL_BATHY_ESTIMATION_CLS[local_estimator_code]
    except KeyError:
        msg = f'{local_estimator_code} is not a supported local bathymetry estimation method.'
        raise NotImplementedError(msg)
    return local_bathy_estimator_cls
