# -*- coding: utf-8 -*-
""" Class encapsulating operations on the radon transform of an image for waves processing

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 4 mars 2021
"""
from functools import lru_cache
from typing import Optional  # @NoMove

import numpy as np  # @NoMove

from ..generic_utils.quantized_directions import (linear_directions,
                                                  DEFAULT_ANGLE_MIN, DEFAULT_ANGLE_MAX)
from ..generic_utils.symmetric_radon import symmetric_radon
from .sinograms_array import SinogramsArray
from .waves_image import WavesImage


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
class WavesRadon(SinogramsArray):
    """ Class handling the Radon transform of some image.
    """

    def __init__(self, image: WavesImage, selected_directions: Optional[np.ndarray] = None,
                 directions_step: float = 1., weighted: bool = False) -> None:
        """ Constructor

        :param image: a 2D array containing an image
        :param selected_directions: a set of directions onto which the radon transform must be
                                    provided. If unspecified, all integre angles between -180° and
                                    +180° are considered.
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

        # TODO: Quantize directions when selected_directions is provided?
        if selected_directions is None:
            selected_directions = linear_directions(DEFAULT_ANGLE_MIN, DEFAULT_ANGLE_MAX,
                                                    directions_step)

        radon_transform_array = self._compute(image.pixels, weighted, selected_directions)

        super().__init__(radon_transform_array, selected_directions, directions_step)

    @staticmethod
    def _compute(pixels: np.ndarray, weighted: bool, selected_directions: np.ndarray) -> np.ndarray:
        """ Compute the radon transform of the image over a set of directions
        """
        # FIXME: quantization may imply that radon transform is not computed on stored directions
        radon_transform_array = symmetric_radon(pixels, theta=selected_directions)

        if weighted:
            weights = sinogram_weights(radon_transform_array.shape[0])
            for direction in range(radon_transform_array.shape[1]):
                radon_transform_array[:, direction] = (
                    radon_transform_array[:, direction] / weights)

        return radon_transform_array

    @property
    def spectrum_wave_numbers(self) -> np.ndarray:
        """ :returns: wave numbers for each sample of the positive part of the FFT of a direction.
        """
        return np.arange(0, self.sampling_frequency / 2, self.sampling_frequency / self.nb_samples)

    def radon_augmentation(self, factor_augmented_radon: float) -> SinogramsArray:
        """ Augment the resolution of the radon transform along the sinogram direction

        :param factor_augmented_radon: factor of the resolution augmentation.
        :return: a new SinogramsArray object with augmented resolution
        """
        radon_transform_augmented_array = np.empty(
            (int(self.nb_samples / factor_augmented_radon), self.nb_directions))
        for index, direction in enumerate(self.directions):
            sinogram = self.get_sinogram(direction)
            radon_transform_augmented_array[:, index] = sinogram.interpolate(
                factor_augmented_radon)
        return SinogramsArray(radon_transform_augmented_array,
                              self.directions, self.quantization_step)

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
        radon_excerpt = self.get_as_array(directions)

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
