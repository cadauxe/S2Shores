# -*- coding: utf-8 -*-
""" Selection of the desired local bathymetry estimator

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 18/06/2021
"""
from typing import List, Dict, Type, TYPE_CHECKING  # @NoMove

from ..image_processing.waves_image import WavesImage
from .local_bathy_estimator import LocalBathyEstimator
from .spatial_correlation_bathy_estimator import SpatialCorrelationBathyEstimator
from .spatial_dft_bathy_estimator import SpatialDFTBathyEstimator
from .temporal_correlation_bathy_estimator import TemporalCorrelationBathyEstimator, \
    TemporalCorrelationBathyEstimatorDebug

if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport

# Dictionary of classes to be instanciated for each local bathymetry estimator
LOCAL_BATHY_ESTIMATION_CLS: Dict[str, Type[LocalBathyEstimator]]
LOCAL_BATHY_ESTIMATION_CLS_DEBUG: Dict[str, Type[LocalBathyEstimator]]
LOCAL_BATHY_ESTIMATION_CLS = {'SPATIAL_DFT': SpatialDFTBathyEstimator,
                              'TEMPORAL_CORRELATION': TemporalCorrelationBathyEstimator,
                              'SPATIAL_CORRELATION': SpatialCorrelationBathyEstimator}

LOCAL_BATHY_ESTIMATION_CLS_DEBUG = {'TEMPORAL_CORRELATION': TemporalCorrelationBathyEstimatorDebug}


def local_bathy_estimator_factory(images_sequence: List[WavesImage],
                                  estimator: 'BathyEstimator') -> LocalBathyEstimator:
    """ builds an instance of a local bathymetry estimator class using the estimator estimator code

    :returns: an instance of a local bathymetry estimator suitable for running estimation
    """
    local_bathy_estimator_cls = get_local_bathy_estimator_cls(estimator.waveparams.WAVE_EST_METHOD,estimator.waveparams.DEBUG_MODE)
    return local_bathy_estimator_cls(images_sequence, estimator)


def get_local_bathy_estimator_cls(local_estimator_code: str, debug_mode: bool) -> Type[LocalBathyEstimator]:
    """ return the local bathymetry estimator class corresponding to a given estimator code

    :returns: the local bathymetry estimator class corresponding to a given estimator code
    :raises NotImplementedError: when the requested bathymetry estimator is unknown
    """
    try:
        if debug_mode:
            local_bathy_estimator_cls = LOCAL_BATHY_ESTIMATION_CLS_DEBUG[local_estimator_code]
        else:
            local_bathy_estimator_cls = LOCAL_BATHY_ESTIMATION_CLS[local_estimator_code]
    except KeyError:
        if debug_mode:
            msg = f'{local_estimator_code} is not a supported local bathymetry estimation method with debug mode'
        else:
            msg = f'{local_estimator_code} is not a supported local bathymetry estimation method.'
        raise NotImplementedError(msg)
    return local_bathy_estimator_cls
