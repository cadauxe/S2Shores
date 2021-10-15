# -*- coding: utf-8 -*-
""" Class encapsulating operations on the radon transform of an image for waves processing

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 4 mars 2021
"""
from typing import Optional, Dict, Tuple, Any  # @NoMove

import numpy as np  # @NoMove

from ..generic_utils.directional_array import DirectionalArray
from ..generic_utils.signal_utils import get_unity_roots

from .waves_sinogram import WavesSinogram, SignalProcessingFilters


SinogramsSetType = Dict[float, WavesSinogram]


class SinogramsArray(DirectionalArray):
    """ Class holding the sinograms of a Radon transform over a set of directions without
    knowledge of the image
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sampling_frequency = 0.
        # FIXME: this is a copy of the sinograms, not pointers to sinograms views
        # FIXME: this is almost redundant now
        self._sinograms: Optional[SinogramsSetType] = None

    # +++++++++++++++++++ Sinograms management part (could go in another class) +++++++++++++++++++

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

    @property
    def sinograms(self) -> SinogramsSetType:
        """ the sinograms of the Radon transform as a dictionary indexed by the directions

        :returns: the sinograms of the Radon transform as a dictionary indexed by the directions
        """
        if self._sinograms is None:
            self._sinograms = self._get_sinograms_as_dict()
        return self._sinograms

    # FIXME: this method should be removed or generalized
    def _get_sinograms_as_dict(self, directions: Optional[np.ndarray] = None) -> SinogramsSetType:
        """ returns the sinograms of the Radon transform as a dictionary indexed by the directions

        :param directions: the set of directions which must be provided in the output dictionary.
                           When unspecified, all the directions of the Radon transform are returned.
        :returns: the sinograms of the Radon transform as a dictionary indexed by the directions
        """
        directions = self.directions if directions is None else directions
        sinograms_dict: SinogramsSetType = {}
        for direction in directions:
            sinograms_dict[direction] = self.get_sinogram(direction)
        return sinograms_dict

    def get_sinogram(self, direction: float) -> WavesSinogram:
        """ returns a new sinogram taken from the Radon transform at some direction

        :param direction: the direction of the requested sinogram.
        :returns: the sinogram of the Radon transform along the requested direction
        """
        return WavesSinogram(self[direction])

    # +++++++++++++++++++ Sinograms processing part +++++++++++++++++++

    def apply_filters(self, processing_filters: SignalProcessingFilters) -> None:
        """ Apply filters on the sinograms in place

        :param processing_filters: A list of functions together with their parameters to apply
                                   sequentially to each sinogram.
        """
        for direction in self.directions:
            # TODO: add an apply_filters to Sinogram and use it
            sinogram = self.get_sinogram(direction).values
            for processing_filter, filter_parameters in processing_filters:
                sinogram = np.array([processing_filter(sinogram, *filter_parameters)]).T
            self[direction] = sinogram

    # TODO: insert into compute_sinograms_dft
    @staticmethod
    def _dft_interpolated(radon_excerpt: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
        """ Computes the 1D dft of a 2D signal along the columns using specific sampling frequencies

        :param radon_excerpt: a 2D signal
        :param frequencies: a set of unevenly spaced frequencies at which the DFT must be computed
        :returns: a 2D array with the DFTs of the selected input columns, stored as contiguous
                  columns
        """
        nb_columns = radon_excerpt.shape[1]
        signal_dft_1d = np.empty((frequencies.size, nb_columns), dtype=np.complex128)

        # FIXME: used to interpolate spectrum, but seems incorrect. Use zero padding instead ?
        unity_roots = get_unity_roots(frequencies, radon_excerpt.shape[0])
        for i in range(nb_columns):
            signal_dft_1d[:, i] = np.dot(unity_roots, radon_excerpt[:, i])
        return signal_dft_1d

    def compute_sinograms_dfts(self,
                               directions: Optional[np.ndarray] = None,
                               kfft: Optional[np.ndarray] = None) -> None:
        """ Computes the fft of the radon transform along the projection directions

        :param directions: the set of directions for which the sinograms DFT must be computed
        :param kfft: the set of wavenumbers to use for sampling the DFT. If None, standard DFT
                     sampling is done.
        """
        frequencies = None if kfft is None else kfft / self.sampling_frequency
        # If no selected directions, DFT will be computed on all directions
        directions = self.directions if directions is None else directions
        # Build array on which the dft will be computed
        radon_excerpt, directions = self.get_as_arrays(directions)

        if frequencies is None:
            # Compute standard DFT along the column axis and keep positive frequencies only
            nb_positive_coeffs = int(np.ceil(radon_excerpt.shape[0] / 2))
            radon_dft_1d = np.fft.fft(radon_excerpt, axis=0)
            result = radon_dft_1d[0:nb_positive_coeffs, :]
        else:
            result = self._dft_interpolated(radon_excerpt, frequencies)
        # Store individual 1D DFTs in sinograms
        for sino_index in range(result.shape[1]):
            sinogram = self.sinograms[directions[sino_index]]
            sinogram.dft = result[:, sino_index]

    def get_sinograms_dfts(self, directions: Optional[np.ndarray] = None) -> np.ndarray:
        """ Retrieve the current DFT of the sinograms in some directions. If DFTs does not exist
        they are computed using standard frequencies.

        :param directions: the directions of the requested sinograms.
                           Defaults to all the Radon transform directions if unspecified.
        :return: the sinograms DFTs for the specified directions or for all directions
        """
        directions = self.directions if directions is None else directions
        fft_sino_length = self.sinograms[directions[0]].dft.shape[0]
        result = np.empty((fft_sino_length, len(directions)), dtype=np.complex128)
        for result_index, sinogram_index in enumerate(directions):
            sinogram = self.sinograms[sinogram_index]
            result[:, result_index] = sinogram.dft
        return result

    def get_sinograms_mean_power(self, directions: Optional[np.ndarray] = None) -> np.ndarray:
        """ Retrieve the mean power of the sinograms in some directions.

        :param directions: the directions of the requested sinograms.
                           Defaults to all the Radon transform directions if unspecified.
        :return: the sinograms mean powers for the specified directions or for all directions
        """
        directions = self.directions if directions is None else directions
        return np.array([self.sinograms[direction].mean_power for direction in directions])

    def get_sinograms_variances(self,
                                processing_filters: Optional[SignalProcessingFilters] = None,
                                directions: Optional[np.ndarray] = None) -> np.ndarray:
        """ Return array of variance of each sinogram

        :param processing_filters: a set a filters to apply on sinograms before computing variance.
                                   Sinograms are left unmodified
        :param directions: the directions of the requested sinograms.
                           Defaults to all the Radon transform directions if unspecified.
        :return: variances of the sinograms
        """
        directions = self.directions if directions is None else directions
        sinograms_variances = np.empty(len(directions), dtype=np.float64)
        for result_index, direction in enumerate(directions):
            sinogram = self.sinograms[direction]
            if processing_filters is not None:
                for filter_name, filter_parameters in processing_filters:
                    sinogram = WavesSinogram(
                        np.array(
                            [filter_name(sinogram.values, *filter_parameters)]).T)
            sinograms_variances[result_index] = sinogram.variance
        return sinograms_variances

    def get_sinogram_maximum_variance(self,
                                      processing_filters: Optional[SignalProcessingFilters] = None,
                                      directions: Optional[np.ndarray] = None) \
            -> Tuple[WavesSinogram, float, np.ndarray]:
        """ Find the sinogram with maximum variance among the set of sinograms on some directions,
        and returns it together with the direction value.

        :param processing_filters: a set a filter to apply on sinograms before computing maximum
                                   variance. Sinograms are left unmodified
        :param directions: a set of directions to look for maximum variance sinogram. If None, all
                           the directions in the radon transform are considered.
        :returns: the sinogram of maximum variance together with the corresponding direction.
        """
        directions = self.directions if directions is None else directions
        variances = self.get_sinograms_variances(processing_filters, directions)
        index_max_variance = np.argmax(variances)
        return self.sinograms[directions[index_max_variance]], directions[
            index_max_variance], variances
