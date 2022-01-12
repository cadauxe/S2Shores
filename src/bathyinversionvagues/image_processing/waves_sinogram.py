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

from ..generic_utils.numpy_utils import HashableNdArray
from ..generic_utils.signal_utils import get_unity_roots


SignalProcessingFilters = List[Tuple[Callable, List[Any]]]


# TODO: make this class derive from a "1D_signal" class which would implement signal processing ?
# This class would gather several functions from signal_utils.
# TODO: introduce sampling_frequency/resolution inside the sinogram itself ?
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
        self._dft = np.array([])
        self._interpolated_dft = np.array([])
        self._interpolated_dft_frequencies = np.array([])

    def interpolate(self, factor: float) -> 'WavesSinogram':
        """ Compute an augmented version of the sinogram, by interpolation with some factor.

        :param factor: fraction of the sinogram sampling step for which new samples has to be evenly
                       interpolated.
        :returns: the interpolated sinogram.
        """
        new_axis = np.linspace(0, self.size - 1, int(self.size / factor))
        current_axis = np.linspace(0, self.size - 1, self.size)
        return WavesSinogram(np.interp(new_axis, current_axis, self.values))

    @property
    def dft(self) -> np.ndarray:
        """ :returns: the current DFT of the sinogram. If it does not exists, it is computed
        from the sinogram using standard frequencies.
        """
        if self._dft.size == 0:
            self.compute_dft()
        return self._dft

    @dft.setter
    def dft(self, dft_values: np.ndarray) -> None:
        self._dft = dft_values

    @property
    def interpolated_dft(self) -> np.ndarray:
        """ :returns: the current DFT of the sinogram. If it does not exists, an exception is raised

        :raises ValueError: if the interpolated DFT does not exist
        """
        if self._interpolated_dft.size == 0:
            raise ValueError('Interpolated DFT does not exist')
        return self._interpolated_dft

    @interpolated_dft.setter
    def interpolated_dft(self, dft_values: np.ndarray) -> None:
        self._interpolated_dft = dft_values

    @property
    def interpolated_dft_frequencies(self) -> np.ndarray:
        """ :returns: the frequencies of the current interpolated DFT of the sinogram.
        If this DFT does not exists, an exception is raised

        :raises ValueError: if the interpolated DFT does not exist
        """
        if self._interpolated_dft_frequencies.size == 0:
            raise ValueError('Interpolated DFT does not exist')
        return self._interpolated_dft_frequencies

    @interpolated_dft_frequencies.setter
    def interpolated_dft_frequencies(self, dft_frequencies_values: np.ndarray) -> None:
        self._interpolated_dft_frequencies = dft_frequencies_values

    def symmeterize(self) -> 'WavesSinogram':
        """ :returns: a new WavesSinogram which is the symmetric version of this one.
        """
        symmetric_sinogram = WavesSinogram(np.flip(self.values))
        # TODO: fill in the sinogram properties based on the current values
        return symmetric_sinogram

    def compute_dft(self, frequencies: Optional[HashableNdArray] = None) -> None:
        """ Computes the DFT of the sinogram

        :param frequencies: a set of frequencies at which the DFT must be computed. If None standard
        frequencies are computed.
        """
        if frequencies is None:
            nb_positive_coeffs = int(np.ceil(self.size / 2))
            self._dft = np.fft.fft(self.values)[0:nb_positive_coeffs]
        else:
            # FIXME: used to interpolate spectrum, but seems incorrect. Use zero padding instead ?
            unity_roots = None if frequencies is None else get_unity_roots(frequencies, self.size)
            self._interpolated_dft = np.dot(unity_roots, self.values)
            self._interpolated_dft_frequencies = frequencies.unwrap()

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

    def apply_filters(self, processing_filters: SignalProcessingFilters) -> 'WavesSinogram':
        """ Apply a set of filters on the sinogram.

        :param processing_filters: A list of functions together with their parameters to apply
                                   sequentially to the sinogram.
        :returns: the filtered sinogram
        """
        sinogram_values = self.values
        for processing_filter, filter_parameters in processing_filters:
            sinogram_values = processing_filter(sinogram_values, *filter_parameters)
        return WavesSinogram(sinogram_values)
