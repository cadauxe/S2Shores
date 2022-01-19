# -*- coding: utf-8 -*-
""" Class encapsulating operations on the radon transform of an image for waves processing

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 4 mars 2021
"""
from functools import lru_cache

from typing import Optional, List  # @NoMove

import numpy as np  # @NoMove


from ..generic_utils.symmetric_radon import symmetric_radon
from .sinograms import Sinograms
from .waves_image import WavesImage
from .waves_sinogram import WavesSinogram


DEFAULT_ANGLE_MIN = -180.
DEFAULT_ANGLE_MAX = 180.
DEFAULT_ANGLE_STEP = 1.


def linear_directions(angle_min: float, angle_max: float, directions_step: float) -> np.ndarray:
    return np.linspace(angle_min, angle_max,
                       int((angle_max - angle_min) / directions_step),
                       endpoint=False)


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


class WavesRadon(Sinograms):
    """ Class handling the Radon transform of some image.
    """

    def __init__(self, image: WavesImage, selected_directions: Optional[np.ndarray] = None,
                 directions_quantization: Optional[float] = None, weighted: bool = False) -> None:
        """ Constructor

        :param image: a 2D array containing an image
        :param selected_directions: a set of directions onto which the radon transform must be
                                    provided. If unspecified, all integre angles between -180° and
                                    +180° are considered.
        :param directions_quantization: the step to use for quantizing direction angles, for
                                        indexing purposes. Direction quantization is such that the
                                        0 degree direction is used as the origin, and any direction
                                        angle is transformed to the nearest quantized angle for
                                        indexing that direction in the radon transform.
        :param weighted: a flag specifying if the radon transform must be weighted by a 1/cos(d)
                         weighting function
        """
        super().__init__(image.sampling_frequency, directions_quantization)

        self.pixels = image.pixels.copy()

        # TODO: Quantize directions when selected_directions is provided?
        if selected_directions is None:
            selected_directions = linear_directions(DEFAULT_ANGLE_MIN, DEFAULT_ANGLE_MAX,
                                                    DEFAULT_ANGLE_STEP)

        radon_transform = symmetric_radon(self.pixels, theta=selected_directions)

        sinograms: List[WavesSinogram] = []
        for index, _ in enumerate(selected_directions):
            sinogram = WavesSinogram(radon_transform[:, index])
            sinograms.append(sinogram)

        self.insert_sinograms(sinograms, selected_directions)
