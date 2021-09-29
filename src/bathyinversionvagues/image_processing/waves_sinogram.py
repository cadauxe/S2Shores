# -*- coding: utf-8 -*-
""" Class encapsulating the "sinogram" of a radon transform, i.e. the radon transform in a given
 direction


:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 4 mars 2021
"""
from typing import Optional, List, Tuple, Callable, Any  # @NoMove
import numpy as np

from ..generic_utils.signal_filters import filter_mean
from ..generic_utils.signal_utils import get_unity_roots, DFT_fr


SignalProcessingFilters = List[Tuple[Callable, List[Any]]]


# TODO: make this class derive from a "1D_signal" class which would implement signal processing ?
# This class would gather several functions from shoreutils.
# TODO: introduce direction inside the sinogram itself ?
class WavesSinogram:
    """ Class handling a sinogram (the component of a Radon transform in some direction)
    """

    def __init__(self, sinogram: np.ndarray) -> None:
        """ Constructor

        :param sinogram: a 1D array containing the sinogram values
        """
        self.sinogram = sinogram
        self.nb_samples = sinogram.shape[0]
        self._dft = None

    def interpolate(self, factor: float) -> np.ndarray:
        """ Compute an augmented version of the sinogram, by interpolation with some factor.

        :param factor: fraction of the sinogram sampling step for which new samples has to be evenly
                       interpolated.
        :returns: the interpolated sinogram as a 1D array.
        """
        new_axis = np.linspace(0, self.nb_samples - 1, int(self.nb_samples / factor))
        current_axis = np.linspace(0, self.nb_samples - 1, self.nb_samples)
        return np.interp(new_axis, current_axis, self.sinogram)

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

    def compute_dft(self, frequencies: Optional[np.ndarray] = None) -> np.ndarray:
        """ Computes the DFT of the sinogram

        :param frequencies: a set of unevenly spaced frequencies at which the DFT must be computed
        :returns: the DFT of the sinogram
        """
        if frequencies is None:
            nb_positive_coeffs = int(np.ceil(self.nb_samples / 2))
            result = np.fft.fft(self.sinogram)[0:nb_positive_coeffs]
        else:
            unity_roots = get_unity_roots(frequencies, self.nb_samples)
            result = DFT_fr(self.sinogram, unity_roots)
        return result

    def filter_mean(self, kernel_size: int) -> np.ndarray:
        """ Apply a mean filter on the sinogram

        :param kernel_size: the number of samples to consider in the filtering window
        :returns: the mean filtered sinogram
        """
        array = np.ndarray.flatten(self.sinogram)
        return filter_mean(array, kernel_size)

    @property
    def energy(self) -> float:
        """ :returns: the energy of the sinogram
        """
        return float(np.sum(self.sinogram * self.sinogram))

    @property
    def mean_power(self) -> float:
        """ :returns: the mean power of the sinogram
        """
        return self.energy / len(self.sinogram)

    @property
    def variance(self) -> float:
        """ :returns: the variance of the sinogram
        """
        return float(np.var(self.sinogram))
