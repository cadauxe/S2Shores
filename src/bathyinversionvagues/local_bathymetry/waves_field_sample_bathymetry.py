# -*- coding: utf-8 -*-
""" Class handling the information describing a waves field sample..

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 6 mars 2021
"""
import numpy as np

from ..bathy_physics import period_offshore, funLinearC_k
from .waves_field_sample_dynamics import WavesFieldSampleDynamics

KNOWN_DEPTH_ESTIMATION_METHODS = ['LINEAR']


class WavesFieldSampleBathymetry(WavesFieldSampleDynamics):
    """ This class encapsulates the bathymetric information for a given sample.

    It inherits from WavesFieldSampleDynamics and defines specific attributes related to the
    bathymetry for that sample..
    """

    def __init__(self, depth_precision: float, gravity: float,
                 depth_estimation_method: str) -> None:
        """ Constructor

        :param depth_precision: precision (in meters) to be used for depth estimation
        :param gravity: the acceleration of gravity to use (m.s-2)
        :param depth_estimation_method: the name of the depth estimation method to use
        :raises NotImplementedError: when the depth estimation method is unsupported
        """
        if depth_estimation_method not in KNOWN_DEPTH_ESTIMATION_METHODS:
            msg = f'{depth_estimation_method} is not a supported depth estimation method.'
            msg += f' Must be one of {KNOWN_DEPTH_ESTIMATION_METHODS}'
            raise NotImplementedError(msg)

        super().__init__()

        self._gravity = gravity
        self._depth_precision = depth_precision
        self._depth_estimation_method = depth_estimation_method

    @property
    def depth(self) -> float:
        """ The estimated depth

        :returns: The depth (m)
        :raises AttributeError: when the depth estimation method is not supported
        """
        if self._depth_estimation_method == 'LINEAR':
            estimated_depth = funLinearC_k(self.wavenumber, self.celerity,
                                           self._depth_precision, self._gravity)
        else:
            msg = 'depth attribute undefined when depth estimation method is not supported'
            raise AttributeError(msg)
        return estimated_depth

    @property
    def linearity(self) -> float:
        """ :returns: a linearity indicator for depth estimation (unitless) """
        return self.wavenumber * 2 * np.pi * (self.celerity ** 2) / self._gravity

    @property
    def period_offshore(self) -> float:
        """ :returns: The offshore period (s) """
        return period_offshore(self.wavenumber, self._gravity)

    def __str__(self) -> str:
        result = WavesFieldSampleDynamics.__str__(self)
        result += f'\ndepth: {self.depth:5.2f} (m)   gamma: {self.linearity:5.2f}  '
        result += f' offshore period: {self.period_offshore:5.2f} (s)'
        return result
