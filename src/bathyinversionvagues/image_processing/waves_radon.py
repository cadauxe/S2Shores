# -*- coding: utf-8 -*-
"""
module -- Class encapsulating operations on the radon transform of an image for waves processing


:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 4 mars 2021
"""
from functools import lru_cache
from typing import Optional, Dict  # @NoMove

import numpy as np  # @NoMove

from ..generic_utils.directional_array import (DirectionalArray, linear_directions,
                                               DEFAULT_ANGLE_MIN, DEFAULT_ANGLE_MAX)
from ..generic_utils.symmetric_radon import symmetric_radon
from ..waves_exceptions import NoRadonTransformError

from .shoresutils import DFT_fr, get_unity_roots
from .waves_image import WavesImage
from .waves_sinogram import WavesSinogram


@lru_cache()
def sinogram_weights(nb_samples: int) -> np.ndarray:
    weights = np.cos(np.linspace(-np.pi / 2., (np.pi / 2.), nb_samples))
    weights[0] = weights[1]
    weights[-1] = weights[-2]
    return weights


# TODO: finalize directions indices removal
class WavesRadon:
    def __init__(self, image: WavesImage, directions_step: float = 1.,
                 weighted: bool = False) -> None:
        """ Constructor

        :param image: a 2D array containing an image over water
        :param directions_step: the step to use for quantizing direction angles, for indexing
                                purposes. Direction quantization is such that the 0 degree direction
                                is used as the origin, and any direction angle is transformed to the
                                nearest quantized angle for indexing that direction in the radon
                                transform.
        """
        self.pixels = image.pixels
        self.sampling_frequency = image.sampling_frequency
        self.nb_samples = min(self.pixels.shape)

        self.directions_step = directions_step

        self._sinograms: Dict[float, WavesSinogram] = {}
        self._radon_transform: Optional[DirectionalArray] = None

        self._weights = sinogram_weights(self.nb_samples) if weighted else None

    @property
    def directions(self) -> np.ndarray:
        """ :return: the directions over which the radon transform has been / must be computed """
        if self._radon_transform is None:
            raise AttributeError('No radon transform computed yet')
        return self._radon_transform.directions

    @property
    def nb_directions(self) -> int:
        """ :return: the number of directions defined for this radon transform"""
        return self.radon_transform.nb_directions

    @property
    def radon_transform(self) -> Optional[DirectionalArray]:
        """
        :returns: the radon transform of the image for the currently defined set of directions
        :raises AttributeError: if the directions have not been specified yet
        """
        return self._radon_transform

    @property
    def spectrum_wave_numbers(self) -> np.ndarray:
        """ :returns: wave numbers for each sample of the positive part of the FFT of a direction.
        """
        return np.arange(0, self.sampling_frequency / 2, self.sampling_frequency / self.nb_samples)

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


# +++++++++++++++++++ Sinograms management part (could go in another class) +++++++++++++++++++

    @property
    def sinograms(self) -> Dict[float, WavesSinogram]:
        if not self._sinograms.keys():
            if self._radon_transform is None:
                raise NoRadonTransformError()
            self._sinograms = self.get_sinograms_as_dict()
        return self._sinograms

    def get_sinograms_as_dict(self,
                              directions: Optional[np.ndarray] = None) -> Dict[float, WavesSinogram]:
        directions = self.directions if directions is None else directions
        sinograms_dict: Dict[float, WavesSinogram] = {}
        if self.radon_transform is not None:
            for direction in self.directions:
                sinograms_dict[direction] = self.get_sinogram(direction)
        return sinograms_dict

    def get_sinogram(self, direction: float) -> WavesSinogram:
        if self._radon_transform is None:
            raise NoRadonTransformError()
        return WavesSinogram(self._radon_transform.values_for(direction), self.sampling_frequency)

    def compute_sinograms_dfts(self,
                               directions: Optional[np.ndarray] = None,
                               kfft: Optional[np.ndarray] = None) -> None:
        """ Computes the fft of the radon transform along the projection directions

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
            result = self._dft_interpolated(radon_excerpt, self.sampling_frequency, kfft)
        # Store individual 1D DFTs in sinograms
        for sino_index in range(result.shape[1]):
            sinogram = self.sinograms[directions[sino_index]]
            sinogram.dft = result[:, sino_index]

    # Make a function close to DFT_fr
    @staticmethod
    def _dft_interpolated(signal_2d: np.ndarray, column_sampling_frequency: float,
                          kfft: np.ndarray) -> np.ndarray:
        """ Computes the 1D dft of a 2D signal along the columns using specific sampling frequencies

        :param signal_2d: a 2D signal
        :param column_sampling_frequency: the sampling frequency along the signal columns
        :param kfft: a table of unevenly spaced frequencies at which the DFT must be computed
        :param column_indices: the column indices on which the DFT is required

        :returns: a 2D array with the DFTs of the selected input columns are stored as contiguous
                  columns
        """
        nb_columns = signal_2d.shape[1]
        signal_dft_1d = np.empty((kfft.size, nb_columns), dtype=np.complex128)

        unity_roots = get_unity_roots(signal_2d.shape[0], kfft, column_sampling_frequency)
        for i in range(nb_columns):
            signal_dft_1d[:, i] = DFT_fr(signal_2d[:, i], unity_roots)
        return signal_dft_1d

    def get_sinograms_dfts(self, directions: Optional[np.ndarray] = None) -> np.ndarray:
        directions = self.directions if directions is None else directions
        fft_sino_length = self.sinograms[directions[0]].dft.shape[0]
        result = np.empty((fft_sino_length, len(directions)), dtype=np.complex128)
        for result_index, sinogram_index in enumerate(directions):
            sinogram = self.sinograms[sinogram_index]
            result[:, result_index] = sinogram.dft
        return result

    def get_sinograms_mean_power(self, directions: Optional[np.ndarray] = None) -> np.ndarray:
        directions = self.directions if directions is None else directions
        sinograms_powers = np.empty(len(directions), dtype=np.float64)
        for result_index, sinogram_index in enumerate(directions):
            sinogram = self.sinograms[sinogram_index]
            sinograms_powers[result_index] = sinogram.mean_power
        return sinograms_powers

    def get_sinogram_variance(self, directions: Optional[np.ndarray] = None) -> np.ndarray:
        directions = self.directions if directions is None else directions
        sinograms_variances = np.empty(len(directions), dtype=np.float64)
        for result_index, sinogram_index in enumerate(directions):
            sinogram = self.sinograms[sinogram_index]
            sinograms_variances[result_index] = sinogram.variance
        return sinograms_variances
