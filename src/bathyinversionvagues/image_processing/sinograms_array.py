# -*- coding: utf-8 -*-
""" Class encapsulating operations on the radon transform of an image for waves processing

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 4 mars 2021
"""
from typing import Optional, Any  # @NoMove

import numpy as np  # @NoMove

from ..generic_utils.directional_array import DirectionalArray
from .sinograms_dict import SinogramsDict
from .waves_sinogram import WavesSinogram


class SinogramsArray(DirectionalArray):
    """ Class holding the sinograms of a Radon transform over a set of directions without
    knowledge of the image
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sinograms: Optional[SinogramsDict] = None

    # +++++++++++++++++++ Sinograms management part (could go in another class) +++++++++++++++++++
    @property
    def sinograms(self) -> SinogramsDict:
        """ the sinograms of the Radon transform as a dictionary indexed by the directions

        :returns: the sinograms of the Radon transform as a dictionary indexed by the directions
        """
        if self._sinograms is None:
            self._sinograms = self.get_sinograms_subset()
        return self._sinograms

    def get_sinograms_subset(self, directions: Optional[np.ndarray] = None) -> SinogramsDict:
        """ returns the sinograms of the Radon transform as a dictionary indexed by the directions

        :param directions: the set of directions which must be provided in the output dictionary.
                           When unspecified, all the directions of the Radon transform are returned.
        :returns: the sinograms of the Radon transform as a dictionary indexed by the directions
        """
        directions = self.directions if directions is None else directions
        sinograms_dict = SinogramsDict()
        for direction in directions:
            sinograms_dict[direction] = self.get_sinogram(direction)
        return sinograms_dict

    def get_sinogram(self, direction: float) -> WavesSinogram:
        """ returns a new sinogram taken from the Radon transform at some direction

        :param direction: the direction of the requested sinogram.
        :returns: the sinogram of the Radon transform along the requested direction
        """
        return WavesSinogram(self[direction])
