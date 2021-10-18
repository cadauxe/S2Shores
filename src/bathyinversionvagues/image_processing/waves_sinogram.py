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

from ..generic_utils.signal_utils import get_unity_roots


SignalProcessingFilters = List[Tuple[Callable, List[Any]]]


# TODO: make this class derive from a "1D_signal" class which would implement signal processing ?
# This class would gather several functions from signal_utils.
# TODO: introduce direction inside the sinogram itself ?
class WavesSinogram:
    """ Class handling a sinogram (the component of a Radon transform in some direction)
    """

    def __init__(self, values: np.ndarray) -> None:
        """ Constructor

        :param values: a 1D array containing the sinogram values
        :raises TypeError: when values is not a 1D numpy array
        """
        if not isinstance(values, np.ndarray) or values.ndim != 1:
            raise TypeError('WavesSinogram accepts only a 1D numpy array as argument')
        self.values = values
        self.size = values.size
        self._dft: Optional[np.ndarray] = None

    def interpolate(self, factor: float) -> np.ndarray:
        """ Compute an augmented version of the sinogram, by interpolation with some factor.

        :param factor: fraction of the sinogram sampling step for which new samples has to be evenly
                       interpolated.
        :returns: the interpolated sinogram as a 1D array.
        """
        new_axis = np.linspace(0, self.size - 1, int(self.size / factor))
        current_axis = np.linspace(0, self.size - 1, self.size)
        return np.interp(new_axis, current_axis, self.values[:, 0])

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
            nb_positive_coeffs = int(np.ceil(self.size / 2))
            result = np.fft.fft(self.values)[0:nb_positive_coeffs]
        else:
            # FIXME: used to interpolate spectrum, but seems incorrect. Use zero padding instead ?
            unity_roots = get_unity_roots(frequencies, self.size)
            result = np.dot(unity_roots, self.values)
        return result

    @property
    def energy(self) -> float:
        """ :returns: the energy of the sinogram
        """
        return float(np.sum(self.values * self.values))

    @property
    def mean_power(self) -> float:
        """ :returns: the mean power of the sinogram
        """
        return self.energy / len(self.values)

    @property
    def variance(self) -> float:
        """ :returns: the variance of the sinogram
        """
        return float(np.var(self.values))
