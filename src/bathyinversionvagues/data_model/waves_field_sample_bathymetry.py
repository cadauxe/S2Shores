# -*- coding: utf-8 -*-
""" Class handling the information describing a waves field sample..

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 6 mars 2021
"""
from typing import cast, Tuple

import numpy as np

from ..bathy_physics import (period_offshore, period_low_depth,
                             depth_from_dispersion, linearity_indicator)
from .waves_field_sample_dynamics import WavesFieldSampleDynamics


KNOWN_DEPTH_ESTIMATION_METHODS = ['LINEAR']


class WavesFieldSampleBathymetry(WavesFieldSampleDynamics):
    """ This class encapsulates the bathymetric information for a given sample.

    It inherits from WavesFieldSampleDynamics and defines specific attributes related to the
    bathymetry for that sample..
    """

    def __init__(self, gravity: float, shallow_water_limit: float,
                 depth_estimation_method: str) -> None:
        """ Constructor

        :param gravity: the acceleration of gravity to use (m.s-2)
        :param shallow_water_limit: the depth limit between intermediate and shallow water (m)
        :param depth_estimation_method: the name of the depth estimation method to use
        :raises NotImplementedError: when the depth estimation method is unsupported
        """
        if depth_estimation_method not in KNOWN_DEPTH_ESTIMATION_METHODS:
            msg = f'{depth_estimation_method} is not a supported depth estimation method.'
            msg += f' Must be one of {KNOWN_DEPTH_ESTIMATION_METHODS}'
            raise NotImplementedError(msg)

        super().__init__()

        self._gravity = gravity
        self._shallow_water_limit = shallow_water_limit
        self._depth_estimation_method = depth_estimation_method

    @property
    def gravity(self) -> float:
        """ :returns: the acceleration of the gravity for this waves field sample
        """
        return self._gravity

    @property
    def depth(self) -> float:
        """ The estimated depth

        :returns: The depth (m)
        :raises AttributeError: when the depth estimation method is not supported
        """
        if self._depth_estimation_method == 'LINEAR':
            estimated_depth = depth_from_dispersion(self.wavenumber, self.celerity, self.gravity)
        else:
            msg = 'depth attribute undefined when depth estimation method is not supported'
            raise AttributeError(msg)
        return estimated_depth

    @property
    def linearity(self) -> float:
        """ :returns: a linearity indicator for depth estimation (unitless) """
        return linearity_indicator(self.wavelength, self.celerity, self.gravity)

    def is_linearity_inside(self, linearity_range: Tuple[float, float]) -> bool:
        """ Check if the linearity indicator is within a given range of values.

        :param linearity_range: minimum and maximum values allowed for the linearity indicator
        :returns: True if the linearity indicator is between the minimum and maximum values, False
                  otherwise
        """
        return (not np.isnan(self.linearity) and
                self.linearity >= linearity_range[0] and self.linearity <= linearity_range[1])

    @property
    def period_offshore(self) -> float:
        """ :returns: The offshore period (s) """
        return cast(float, period_offshore(self.wavenumber, self.gravity))

    @property
    def period_low_depth(self) -> float:
        """ :returns:  the period in shallow water (s)
        """
        return cast(float, period_low_depth(self.wavenumber,
                                            self._shallow_water_limit,
                                            self.gravity))

    def __str__(self) -> str:
        result = f'Bathymetry: depth: {self.depth:5.2f} (m)   gamma: {self.linearity:5.2f}  '
        result += f' offshore period: {self.period_offshore:5.2f} (s)'
        result += f' shallow water period: {self.period_low_depth:5.2f} (s)'
        return result
