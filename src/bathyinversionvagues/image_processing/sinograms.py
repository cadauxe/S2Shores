# -*- coding: utf-8 -*-
""" Class encapsulating operations on the radon transform of an image for waves processing

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 4 mars 2021
"""
from typing import Optional, Any, List, Tuple  # @NoMove @UnusedImport

import numpy as np  # @NoMove


from ..generic_utils.signal_utils import get_unity_roots
from .sinograms_dict import SinogramsDict
from .waves_sinogram import WavesSinogram, SignalProcessingFilters


class Sinograms(SinogramsDict):
    """ Class handling a set of sinograms coming from some Radon transform of some image.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._sampling_frequency = 0.
        self.directions_interpolated_dft: Optional[np.ndarray] = None

    @property
    def sampling_frequency(self) -> float:
        """ :return: the sampling frequency of the sinograms """
        return self._sampling_frequency

    @sampling_frequency.setter
    def sampling_frequency(self, frequency: float) -> None:
        self._sampling_frequency = frequency

    @property
    def spectrum_wave_numbers(self) -> np.ndarray:
        """ :returns: wave numbers for each sample of the positive part of the FFT of a direction.
        """
        return np.arange(0, self.sampling_frequency / 2, self.sampling_frequency / self.nb_samples)

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
        filtered_sinograms = Sinograms()
        for direction in self:
            filtered_sinograms[direction] = self[direction].apply_filters(processing_filters)
        return filtered_sinograms

    def compute_sinograms_dfts(self,
                               directions: Optional[np.ndarray] = None,
                               kfft: Optional[np.ndarray] = None) -> None:
        """ Computes the fft of the radon transform along the projection directions

        :param directions: the set of directions for which the sinograms DFT must be computed
        :param kfft: the set of wavenumbers to use for sampling the DFT. If None, standard DFT
                     sampling is done.
        """
        frequencies = None if kfft is None else kfft / self.sampling_frequency
        unity_roots = None if frequencies is None else get_unity_roots(frequencies, self.nb_samples)
        # If no selected directions, DFT is computed on all directions
        directions_to_compute = self.directions if directions is None else directions

        for direction in directions_to_compute:
            if unity_roots is None:
                self[direction].dft = self[direction].compute_dft()
            else:
                self[direction].interpolated_dft = self[direction].compute_dft(unity_roots)

        if unity_roots is not None:
            self.directions_interpolated_dft = directions_to_compute

    def get_sinograms_dfts(self, directions: Optional[np.ndarray] = None,
                           interpolated_dft: bool = False) -> np.ndarray:
        """ Retrieve the current DFT of the sinograms in some directions. If DFTs does not exist
        they are computed using standard frequencies.

        :param directions: the directions of the requested sinograms.
                           Defaults to all the Radon transform directions if unspecified.
        :param interpolated_dft: a flag allowing to select the standard DFT or the interpolated DFT
        :return: the sinograms DFTs for the specified directions or for all directions
        :raises AttributeError: when an interpolated DFT is requested but has not been computed yet.
        """
        if interpolated_dft:
            if self.directions_interpolated_dft is None:
                raise AttributeError('no interpolated DFTs available')
            directions = self.directions_interpolated_dft
            fft_sino_length = self[directions[0]].interpolated_dft.shape[0]
        else:
            directions = self.directions if directions is None else directions
            fft_sino_length = self[directions[0]].dft.shape[0]
        result = np.empty((fft_sino_length, len(directions)), dtype=np.complex128)
        for result_index, direction in enumerate(directions):
            sinogram = self[direction]
            if interpolated_dft:
                result[:, result_index] = sinogram.interpolated_dft
            else:
                result[:, result_index] = sinogram.dft
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
        radon_augmented = Sinograms()
        radon_augmented.quantization_step = self.quantization_step
        radon_augmented.sampling_frequency = self.sampling_frequency / factor_augmented_radon
        radon_augmented.insert_sinograms(radon_transform_augmented_list, self.directions)
        return radon_augmented
