# -*- coding: utf-8 -*-
""" Class encapsulating operations on the radon transform of an image for waves processing

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 4 mars 2021
"""
import copy
from functools import lru_cache
from typing import Optional, Dict, Tuple  # @NoMove

import numpy as np  # @NoMove

from ..generic_utils.directional_array import (DirectionalArray, linear_directions,
                                               DEFAULT_ANGLE_MIN, DEFAULT_ANGLE_MAX)
from ..generic_utils.signal_utils import DFT_fr, get_unity_roots
from ..generic_utils.symmetric_radon import symmetric_radon
from ..waves_exceptions import NoRadonTransformError

from .waves_image import WavesImage
from .waves_sinogram import WavesSinogram, SignalProcessingFilters


SinogramsSetType = Dict[float, WavesSinogram]


@lru_cache()
def sinogram_weights(nb_samples: int) -> np.ndarray:
    """ Computes a cosine weighting function to account for less energy at the extremities of a
    sinogram.

    :param nb_samples: the number of samples in the sinogram (its length)
    :return: half period of cosine with extremities modified to be non-zero

    """
    weights = np.cos(np.linspace(-np.pi / 2., (np.pi / 2.), nb_samples))
    weights[0] = weights[1]
    weights[-1] = weights[-2]
    return weights


# TODO: finalize directions indices removal
class WavesRadon:
    """ Class handling the Radon transform of some image.
    """

    def __init__(self, image: WavesImage, directions_step: float = 1.,
                 weighted: bool = False) -> None:
        """ Constructor

        :param image: a 2D array containing an image over water
        :param directions_step: the step to use for quantizing direction angles, for indexing
                                purposes. Direction quantization is such that the 0 degree direction
                                is used as the origin, and any direction angle is transformed to the
                                nearest quantized angle for indexing that direction in the radon
                                transform.
        :param weighted: a flag specifying if the radon transform must be weighted by a 1/cos(d)
                         weighting function
        """
        self.pixels = image.pixels
        self.sampling_frequency = image.sampling_frequency
        self.nb_samples = min(self.pixels.shape)

        self.directions_step = directions_step

        self._sinograms: SinogramsSetType = {}
        self._radon_transform: Optional[DirectionalArray] = None

        self._weights = sinogram_weights(self.nb_samples) if weighted else None

    @property
    def directions(self) -> np.ndarray:
        """ :return: the directions over which the radon transform has been / must be computed
        :raises AttributeError: if the radon transform has not been computed yet """
        if self._radon_transform is None:
            raise AttributeError('No radon transform computed yet')
        return self._radon_transform.directions

    @property
    def nb_directions(self) -> int:
        """ :return: the number of directions defined for this radon transform"""
        return self.radon_transform.nb_directions

    @property
    def radon_transform(self) -> Optional[DirectionalArray]:
        """ :returns: the radon transform of the image for the currently defined set of directions
        """
        return self._radon_transform

    @property
    def spectrum_wave_numbers(self) -> np.ndarray:
        """ :returns: wave numbers for each sample of the positive part of the FFT of a direction.
        """
        return np.arange(0, self.sampling_frequency / 2, self.sampling_frequency / self.nb_samples)

    def radon_augmentation(self, factor_augmented_radon: float) -> 'WavesRadon':
        """ Augment the resolution of the radon transform along the sinogram direction

        :param factor_augmented_radon: factor of the resolution augmentation.
        :return: a new radon object with augmented resolution
        """
        waves_radon = copy.deepcopy(self)
        radon_transform_augmented_array = np.empty(
            (int(self.nb_samples / factor_augmented_radon), self.nb_directions))
        for index, direction in enumerate(self.directions):
            sinogram = self.get_sinogram(direction)
            radon_transform_augmented_array[:, index] = sinogram.interpolate(
                factor_augmented_radon)
        waves_radon._radon_transform = DirectionalArray(array=radon_transform_augmented_array,
                                                        directions=self.directions,
                                                        directions_step=self.directions_step)
        waves_radon.sampling_frequency = self.sampling_frequency * factor_augmented_radon
        waves_radon.nb_samples = int(self.nb_samples / factor_augmented_radon)
        return waves_radon

    def compute(self, selected_directions: Optional[np.ndarray] = None) -> None:
        """ Compute the radon transform of the image for the currently defined set of directions

        :raises AttributeError: if the directions have not been specified yet
        """
        if selected_directions is None:
            selected_directions = linear_directions(DEFAULT_ANGLE_MIN, DEFAULT_ANGLE_MAX,
                                                    self.directions_step)
        # FIXME: quantization may imply that radon transform is not computed on stored directions
        radon_transform_array = symmetric_radon(self.pixels, theta=selected_directions)
        self._radon_transform = DirectionalArray(array=radon_transform_array,
                                                 directions=selected_directions,
                                                 directions_step=self.directions_step)

        if self._weights is not None and self._radon_transform is not None:
            for direction in range(self.nb_directions):
                self._radon_transform.array[:, direction] = (
                    self._radon_transform.array[:, direction] / self._weights)

    def apply_filter(self, processing_filters: SignalProcessingFilters) -> None:
        """ Apply filters on the image pixels in place

        :param processing_filters: A list of functions together with their parameters to apply
                                   sequentially to the image pixels.
        """
        if self._radon_transform is not None:
            for direction in self.directions:
                sinogram = self.get_sinogram(direction)
                for processing_filter, filter_parameters in processing_filters:
                    sinogram.sinogram = np.array(
                        [processing_filter(sinogram.sinogram.flatten(), *filter_parameters)]).T
                self._radon_transform.set_at_direction(direction, sinogram.sinogram)

    # +++++++++++++++++++ Sinograms management part (could go in another class) +++++++++++++++++++

    @property
    def sinograms(self) -> SinogramsSetType:
        """ the sinograms of the Radon transform as a dictionary indexed by the directions

        :returns: the sinograms of the Radon transform as a dictionary indexed by the directions
        :raises NoRadonTransformError: when the Radon transform has not been computed yet
        """
        if not self._sinograms.keys():
            if self._radon_transform is None:
                raise NoRadonTransformError()
            self._sinograms = self.get_sinograms_as_dict()
        return self._sinograms

    def get_sinograms_as_dict(self, directions: Optional[np.ndarray] = None) -> SinogramsSetType:
        """ returns the sinograms of the Radon transform as a dictionary indexed by the directions

        :param directions: the set of directions which must be provided in the output dictionary.
                           When unspecified, all the directions of the Radon transform are returned.
        :returns: the sinograms of the Radon transform as a dictionary indexed by the directions
        :raises NoRadonTransformError: when the Radon transform has not been computed yet
        """
        directions = self.directions if directions is None else directions
        sinograms_dict: SinogramsSetType = {}
        if self.radon_transform is not None:
            for direction in directions:
                sinograms_dict[direction] = self.get_sinogram(direction)
        return sinograms_dict

    def get_sinograms_as_array(self, directions: Optional[np.ndarray] = None) -> SinogramsSetType:
        """ returns the sinograms of the Radon transform as a np.ndarray of shape (len(sinogram),
        len(direction))

        :param directions: the set of directions which must be provided in the output dictionary.
                           When unspecified, all the directions of the Radon transform are returned.
        :returns: np.ndarray with sinogram on each line
        :raises NoRadonTransformError: when the Radon transform has not been computed yet
        """
        directions = self.directions if directions is None else directions
        sinograms_array = None
        if self.radon_transform is not None:
            for index, direction in enumerate(directions):
                sinogram = self.get_sinogram(direction)
                if sinograms_array is None:
                    sinograms_array = np.empty((len(sinogram.sinogram), len(directions)))
                sinograms_array[:, index] = sinogram.sinogram.flatten()
        return sinograms_array

    def get_sinogram(self, direction: float) -> WavesSinogram:
        """ returns a new sinogram taken from the Radon transform at some direction

        :param direction: the direction of the requested sinogram.
        :returns: the sinogram of the Radon transform along the requested direction
        :raises NoRadonTransformError: when the Radon transform has not been computed yet
        """
        if self._radon_transform is None:
            raise NoRadonTransformError()
        return WavesSinogram(self._radon_transform.values_for(direction))

    def compute_sinograms_dfts(self,
                               directions: Optional[np.ndarray] = None,
                               kfft: Optional[np.ndarray] = None) -> None:
        """ Computes the fft of the radon transform along the projection directions

        :param directions: the set of directions for which the sinograms DFT must be computed
        :param kfft: the set of wavenumbers to use for sampling the DFT. If None, standard DFT
                     sampling is done.
        """
        # If no selected directions, DFT will be computed on all directions
        directions = self.directions if directions is None else directions
        # Build array on which the dft will be computed
        radon_excerpt = self.radon_transform.get_as_array(directions)

        if kfft is None:
            # Compute standard DFT along the column axis and keep positive frequencies only
            nb_positive_coeffs = int(np.ceil(radon_excerpt.shape[0] / 2))
            radon_dft_1d = np.fft.fft(radon_excerpt, axis=0)
            result = radon_dft_1d[0:nb_positive_coeffs, :]
        else:
            frequencies = kfft / self.sampling_frequency
            result = self._dft_interpolated(radon_excerpt, frequencies)
        # Store individual 1D DFTs in sinograms
        for sino_index in range(result.shape[1]):
            sinogram = self.sinograms[directions[sino_index]]
            sinogram.dft = result[:, sino_index]

    # Make a function close to DFT_fr
    @staticmethod
    def _dft_interpolated(signal_2d: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
        """ Computes the 1D dft of a 2D signal along the columns using specific sampling frequencies

        :param signal_2d: a 2D signal
        :param frequencies: a set of unevenly spaced frequencies at which the DFT must be computed
        :returns: a 2D array with the DFTs of the selected input columns, stored as contiguous
                  columns
        """
        nb_columns = signal_2d.shape[1]
        signal_dft_1d = np.empty((frequencies.size, nb_columns), dtype=np.complex128)

        unity_roots = get_unity_roots(frequencies, signal_2d.shape[0])
        for i in range(nb_columns):
            signal_dft_1d[:, i] = DFT_fr(signal_2d[:, i], unity_roots)
        return signal_dft_1d

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
        sinograms_powers = np.empty(len(directions), dtype=np.float64)
        for result_index, sinogram_index in enumerate(directions):
            sinogram = self.sinograms[sinogram_index]
            sinograms_powers[result_index] = sinogram.mean_power
        return sinograms_powers

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
        for result_index, sinogram_index in enumerate(directions):
            sinogram = self.sinograms[sinogram_index]
            if processing_filters is not None:
                for filter, filter_parameters in processing_filters:
                    sinogram = WavesSinogram(
                        np.array(
                            [filter(sinogram.sinogram.flatten(), *filter_parameters)]).T)
            sinograms_variances[result_index] = sinogram.variance
        return sinograms_variances

    def get_sinogram_maximum_variance(self,
                                      processing_filters: Optional[SignalProcessingFilters] = None,
                                      directions: Optional[np.ndarray] = None) \
            -> Tuple[WavesSinogram, float, np.ndarray]:
        """ Find the sinogram with maximum variance among the set of sinograms on some directions,
        and returns it together with the direction value.

        :param preprocessing_filters: a set a filter to apply on sinograms before computing maximum
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
