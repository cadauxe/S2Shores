# -*- coding: utf-8 -*-
""" Class encapsulating operations on the radon transform of an image for waves processing

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
from typing import Any, List, Optional, Tuple  # @NoMove @UnusedImport

import numpy as np  # @NoMove

from ..generic_utils.numpy_utils import HashableNdArray
from .sinograms_dict import SinogramsDict
from .waves_sinogram import SignalProcessingFilters, WavesSinogram


class Sinograms(SinogramsDict):
    """ Class handling a set of sinograms coming from some Radon transform of some image.
    """

    def __init__(self, sampling_frequency: float,
                 directions_quantization: Optional[float] = None) -> None:
        """ Constructor

        :param sampling_frequency: the sampling frequency of the sinograms
        :param directions_quantization: the step to use for quantizing direction angles, for
                                        indexing purposes. Direction quantization is such that the
                                        0 degree direction is used as the origin, and any direction
                                        angle is transformed to the nearest quantized angle for
                                        indexing that direction in the radon transform.
        """
        super().__init__(directions_quantization)

        self._sampling_frequency = sampling_frequency
        self.directions_interpolated_dft: Optional[np.ndarray] = None

    @property
    def sampling_frequency(self) -> float:
        """ :return: the sampling frequency of the sinograms """
        return self._sampling_frequency

    @property
    def spectrum_wave_numbers(self) -> np.ndarray:
        """ :returns: wave numbers for each sample of the positive part of the FFT of a direction.
        """
        nb_positive_coefs = int(np.ceil(self.nb_samples / 2))
        return np.fft.fftfreq(self.nb_samples)[0:nb_positive_coefs] * self.sampling_frequency

    # +++++++++++++++++++ Sinograms processing part +++++++++++++++++++

    def apply_filters(self, processing_filters: SignalProcessingFilters,
                      directions: Optional[np.ndarray] = None) -> 'Sinograms':
        """ Apply filters on the sinograms

        :param processing_filters: A list of functions together with their parameters to apply
                                   sequentially to the selected sinograms.
        :param directions: the directions of the sinograms to filter.
                           Defaults to all the sinograms directions if unspecified.
        :returns: the filtered sinograms
        """
        directions = self.directions if directions is None else directions
        filtered_sinograms = Sinograms(self.sampling_frequency, self.quantization_step)
        for direction in self:
            filtered_sinograms[direction] = self[direction].apply_filters(processing_filters)
        return filtered_sinograms

    def interpolate_sinograms_dfts(self, wavenumbers: np.ndarray,
                                   directions: Optional[np.ndarray] = None) -> None:
        """ Interpolates the dft of the radon transform along the projection directions

        :param wavenumbers: the set of wavenumbers to use for interpolating the DFT.
        :param directions: the set of directions for which the sinograms DFT must be interpolated
        :raises ValueError: when there is no wavenumbers to interpolate from
        """
        # Interpolation can be done only if at least one frequency is requested
        if wavenumbers.size == 0:
            raise ValueError('DFT interpolation requires at least 1 frequency')
        # If no selected directions, DFT is interpolated on all directions
        self.directions_interpolated_dft = self.directions if directions is None else directions
        # Computes normalized frequencies and make hashable
        normalized_frequencies = HashableNdArray(wavenumbers / self.sampling_frequency)
        for direction in self.directions_interpolated_dft:
            self[direction].interpolate_dft(normalized_frequencies)

    def get_sinograms_standard_dfts(self, directions: Optional[np.ndarray] = None) -> np.ndarray:
        """ Retrieve the current DFT of the sinograms in some directions. If DFTs does not exist
        they are computed using standard frequencies.

        :param directions: the directions of the requested sinograms.
                           Defaults to all the Radon transform directions if unspecified.
        :return: the sinograms DFTs for the specified directions or for all directions
        :raises AttributeError: when an interpolated DFT is requested but has not been computed yet.
        """
        directions = self.directions if directions is None else directions
        fft_sino_length = self[directions[0]].dft.size
        result = np.empty((fft_sino_length, len(directions)), dtype=np.complex128)
        for result_index, direction in enumerate(directions):
            sinogram = self[direction]
            result[:, result_index] = sinogram.dft
        return result

    def get_sinograms_interpolated_dfts(self,
                                        directions: Optional[np.ndarray] = None) -> np.ndarray:
        """ Retrieve the interpolated DFTs of the sinograms in some directions.

        :param directions: the directions of the requested sinograms.
                           Defaults to all the Radon transform directions if unspecified.
        :return: the interpolated sinograms DFTs for the specified directions or for all directions
        :raises AttributeError: when an interpolated DFT is requested but has not been computed yet.
        """
        if self.directions_interpolated_dft is None:
            raise AttributeError('no interpolated DFTs available')
        directions = self.directions_interpolated_dft if directions is None else directions
        fft_sino_length = self[directions[0]].interpolated_dft.size
        result = np.empty((fft_sino_length, len(directions)), dtype=np.complex128)
        for result_index, direction in enumerate(directions):
            sinogram = self[direction]
            result[:, result_index] = sinogram.interpolated_dft
        return result

    def get_sinograms_mean_power(self, directions: Optional[np.ndarray] = None) -> np.ndarray:
        """ Retrieve the mean power of the sinograms in some directions.

        :param directions: the directions of the requested sinograms.
                           Defaults to all the Radon transform directions if unspecified.
        :return: the sinograms mean powers for the specified directions or for all directions
        """
        directions = self.directions if directions is None else directions
        return np.array([self[direction].mean_power for direction in directions])

    # FIXME: output cannot be used safely without outputting the directions
    def get_sinograms_variances(self,
                                directions: Optional[np.ndarray] = None) -> np.ndarray:
        """ Return array of variance of each sinogram

        :param directions: the directions of the requested sinograms.
                           Defaults to all the Sinograms directions if unspecified.
        :return: variances of the sinograms
        """
        directions = self.directions if directions is None else directions
        sinograms_variances = np.empty(len(directions), dtype=np.float64)
        for result_index, direction in enumerate(directions):
            sinograms_variances[result_index] = self[direction].variance
        return sinograms_variances

    # FIXME: output cannot be used safely without outputting the directions
    def get_sinograms_energies(self,
                               directions: Optional[np.ndarray] = None) -> np.ndarray:
        """ Return array of energy of each sinogram

        :param directions: the directions of the requested sinograms.
                           Defaults to all the Sinograms directions if unspecified.
        :return: energies of the sinograms
        """
        directions = self.directions if directions is None else directions
        sinograms_energies = np.empty(len(directions), dtype=np.float64)
        for result_index, direction in enumerate(directions):
            sinograms_energies[result_index] = self[direction].energy
        return sinograms_energies

    def get_direction_maximum_variance(self, directions: Optional[np.ndarray] = None) \
            -> Tuple[float, np.ndarray]:
        """ Find the sinogram with maximum variance among the set of sinograms along some
        directions.

        :param directions: a set of directions to look for maximum variance sinogram. If None, all
                           the directions in the Sinograms are considered.
        :returns: the direction of the maximum variance sinogram together with the set of variances.
        """
        directions = self.directions if directions is None else directions
        variances = self.get_sinograms_variances(directions)
        index_max_variance = np.argmax(variances)
        return directions[index_max_variance], variances

    def radon_augmentation(self, factor_augmented_radon: float) -> 'Sinograms':
        """ Augment the resolution of the radon transform along the sinogram direction

        :param factor_augmented_radon: factor of the resolution augmentation.
        :return: a new SinogramsDict object with augmented resolution
        """
        radon_transform_augmented_list: List[WavesSinogram] = []
        for direction in self.directions:
            interpolated_sinogram = self[direction].interpolate(factor_augmented_radon)
            radon_transform_augmented_list.append(interpolated_sinogram)
        radon_augmented = Sinograms(self.sampling_frequency / factor_augmented_radon,
                                    self.quantization_step)
        radon_augmented.insert_sinograms(radon_transform_augmented_list, self.directions)
        return radon_augmented
