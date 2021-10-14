# -*- coding: utf-8 -*-
""" Definition of the DirectionalArray class and associated functions

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 13 Oct 2021
"""
from abc import abstractmethod

from typing import Any, Optional, List

import numpy as np

from .constrained_dict import ConstrainedDict
from .directions_quantizer import DirectionsQuantizer


class QuantizedDirectionsDict(ConstrainedDict):
    """ An abstract dictionary whose keys are directions expressed in degrees and quantized with
    some quantization step. The directions are expressed between -180 and +180 degrees.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sorted_directions: Optional[List[float]] = None
        self._quantizer = DirectionsQuantizer(self.quantization_step)

    @property
    @abstractmethod
    def quantization_step(self) -> float:
        """ :return: the quantization step of the directions defined in this dictionary """

    def constrained_key(self, key: float) -> float:
        # First check if key already accepted
        if key in self.keys():
            return key
        # Key not accepted yet. Check it.
        if not isinstance(key, float):
            raise TypeError('Keys for a QuantizedDirectionsDict must be float.')
        return self._quantizer.quantize_float(key)

    def __setitem__(self, key: float, value: Any) -> None:
        # _sorted_directions attribute must be reset in case a new item enters the dictionary
        self._sorted_directions = None
        ConstrainedDict.__setitem__(self, key, value)

    def __delitem__(self, key: Any) -> None:
        # _sorted_directions attribute must be reset in case an item is deleted from the dictionary
        self._sorted_directions = None
        ConstrainedDict.__delitem__(self, key)

    @property
    def sorted_directions(self) -> List[float]:
        """ :returns: the list of the directions in the dictionary, sorted in ascending order.
        """
        if self._sorted_directions is None:
            self._sorted_directions = sorted(list(self.keys()))
        return self._sorted_directions

    @property
    def directions(self) -> np.ndarray:
        """ :return: the directions defined in this DirectionalArray """
        return np.array(self.sorted_directions)

    @property
    def nb_directions(self) -> int:
        """ :return: the number of directions defined in this DirectionalArray"""
        return len(self)
