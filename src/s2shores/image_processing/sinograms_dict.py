# -*- coding: utf-8 -*-
""" Class encapsulating the sinograms of a Radon transform in a dictionary

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 4 mars 2021

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
from typing import Any, List, Optional, Tuple, cast  # @NoMove

import numpy as np  # @NoMove

from ..generic_utils.quantized_directions_dict import QuantizedDirectionsDict
from .waves_sinogram import WavesSinogram


class SinogramsDict(QuantizedDirectionsDict):
    """ Class holding the sinograms of a Radon transform over a set of directions without
    knowledge of the image
    """

    def __init__(self, directions_quantization: Optional[float] = None) -> None:
        """ Constructor

        :param directions_quantization: the step to use for quantizing direction angles, for
                                        indexing purposes. Direction quantization is such that the
                                        0 degree direction is used as the origin, and any direction
                                        angle is transformed to the nearest quantized angle for
                                        indexing that direction in the radon transform.
        """
        super().__init__(directions_quantization)
        self._nb_samples = -1

    # +++++++++++++++++++ Sinograms management part +++++++++++++++++++

    def __getitem__(self, direction: float) -> Any:
        try:
            sinogram = QuantizedDirectionsDict.__getitem__(self, direction)
        except KeyError:
            # Check to see if the symmetric direction exists in the dictionary.
            # If it exists create the sinogram for the requested sinogram by symmetry
            # If not, raise another KeyError
            opposite_sinogram = QuantizedDirectionsDict.__getitem__(self, direction + 180.)
            sinogram = opposite_sinogram.symmeterize()
            # insert created sinogram in self
            self[direction] = sinogram
        return sinogram

    @property
    def nb_samples(self) -> int:
        """ :return: the length of each Sinogram in this SinogramsDict"""
        return self._nb_samples

    def constrained_value(self, value: Any) -> Any:
        if not isinstance(value, WavesSinogram):
            raise TypeError('Values for a SinogramsDict can only be a WavesSinogram object')
        if self._nb_samples < 0:
            self._nb_samples = value.size
        else:
            if value.size != self._nb_samples:
                msg = 'WavesSinogram objects in a SinogramsDict must have the same size. Expected '
                msg += f'size (from first insert) is {self._nb_samples}, current is {value.size}'
                raise ValueError(msg)
        return value

    def insert_sinograms(self, sinograms: List[WavesSinogram], directions: np.ndarray) -> None:
        """ Insert a set of Sinograms objects, whose directions are provided
        in a 1d array of the same size.

        :param sinograms: the sinograms to insert
        :param directions: the set of directions in degrees associated to each sinogram.
        :raises TypeError: when array or directions have not the right number of dimensions
        :raises ValueError: when the number of dimensions is not consistent with the number of
                            columns in the array.
        """
        # Check that numpy arguments are of the right dimensions and consistent
        if (not isinstance(sinograms, list) or
                any([not isinstance(obj, WavesSinogram) for obj in sinograms])):
            raise TypeError('sinograms argument must be a list of WaveSinogram objects')

        if directions.ndim != 1:
            raise TypeError('directions for a SinogramsDict must be a 1D numpy array')

        if directions.size != len(sinograms):
            raise ValueError('directions size must be equal to the number of columns of the array')

        # FIXME: directions may be quantized.should we check them before insertion
        for direction, sinogram in zip(directions.tolist(), sinograms):
            self[direction] = sinogram
        if self.nb_directions != len(sinograms):
            raise ValueError('dimensions after quantization have not the same number of elements '
                             f'({self.nb_directions}) than the number '
                             f'of sinograms in the list ({len(sinograms)})')

    def get_as_arrays(self,
                      directions: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns a 2D array with the requested directional values as columns

        :param directions: a vector of directions to store in the returned array, in their order
        :returns: a 2D array with the requested directional values as columns
        """
        if directions is None:
            selected_directions = np.array(self.sorted_directions)
        else:
            selected_directions_array = cast(np.ndarray, self.quantizer.quantize(directions))
            selected_directions = np.array(sorted(selected_directions_array.tolist()))

        # Build array by selecting the requested directions
        array_excerpt = np.empty((self.nb_samples, len(selected_directions)))
        for index, direction in enumerate(selected_directions):
            array_excerpt[:, index] = self[direction].values
        return array_excerpt, selected_directions

    def get_sinograms_subset(self, directions: Optional[np.ndarray] = None) -> 'SinogramsDict':
        """ returns the sinograms of the Radon transform as a dictionary indexed by the directions

        :param directions: the set of directions which must be provided in the output dictionary.
                           When unspecified, all the directions of the Radon transform are returned.
        :returns: the sinograms of the Radon transform as a dictionary indexed by the directions
        """
        directions = self.directions if directions is None else directions
        sinograms_dict = SinogramsDict()
        for direction in directions:
            sinograms_dict[direction] = self[direction]
        return sinograms_dict
