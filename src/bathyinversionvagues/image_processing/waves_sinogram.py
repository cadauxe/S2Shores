# -*- coding: utf-8 -*-
""" Class encapsulating the "sinogram" of a radon transform, i.e. the radon transform in a given
 direction


:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 4 mars 2021
"""
from typing import Optional  # @NoMove

import numpy as np

from .shoresutils import get_unity_roots, DFT_fr


class WavesSinogram():
    # TODO: introduce direction inside the sinogram itself ?
    # FIXME: is it a sampling frequency or a sampling period ?
    def __init__(self, sinogram: np.ndarray, sampling_frequency: float) -> None:
        """ Constructor

        :param sinogram: a 1D array containing the sinogram values
        """
        self.sinogram = sinogram
        self.nb_samples = sinogram.shape[0]
        self.sampling_frequency = sampling_frequency
        self._dft = None

    @property
    def dft(self) -> np.ndarray:
        """ :returns: the current DFT of the sinogram. If it does not exists, it is computed
        from the sinogram using standard frequencies.
        """
        if self._dft is None:
            self._dft = self.compute_dft()
        return self._dft

    @dft.setter
    def dft(self, dft_values: np.ndarray) -> None:
        self._dft = dft_values

    def compute_dft(self, kfft: Optional[np.ndarray] = None) -> np.ndarray:
        """ Computes the DFT of the sinogram

        :returns: the DFT of the sinogram
        """
        if kfft is None:
            nb_positive_coeffs = int(np.ceil(self.nb_samples / 2))
            result = np.fft.fft(self.sinogram)[0:nb_positive_coeffs]
        else:
            unity_roots = get_unity_roots(self.nb_samples, kfft, self.sampling_frequency)
            result = DFT_fr(self.sinogram, unity_roots)
        return result

    @property
    def energy(self) -> float:
        return np.sum(self.sinogram * self.sinogram)

    @property
    def mean_power(self) -> float:
        return np.sum(self.sinogram * self.sinogram) / len(self.sinogram)
