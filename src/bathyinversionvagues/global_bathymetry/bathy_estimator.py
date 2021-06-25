# -*- coding: utf-8 -*-
""" Definition of the BathyEstimator abstract class

:author: GIROS Alain
:created: 17/05/2021
"""
from abc import ABC, abstractproperty
from typing import Tuple, Optional, List  # @NoMove

import numpy as np  # @NoMove
from xarray import Dataset  # @NoMove
from munch import Munch

from ..bathy_physics import phi_limits
from ..data_providers.dis_to_shore_provider import DefaultDisToShoreProvider, DisToShoreProvider
from ..data_providers.gravity_provider import ConstantGravityProvider, GravityProvider
from ..image.image_geometry_types import MarginsType, PointType
from ..image.ortho_image import OrthoImage
from ..image.sampled_ortho_image import SampledOrthoImage
from ..local_bathymetry.local_bathy_estimator import WavesFieldsEstimations
from .ortho_bathy_estimator import OrthoBathyEstimator


class BathyEstimator(ABC):
    """ Management of bathymetry computation and parameters on a single product. Computation
    is split in several cartographic tiles, which must be run separately, either in parallel or
    sequentially.
    """

    def __init__(self, image: OrthoImage, wave_params: Munch,
                 nb_subtiles_max: int = 1) -> None:
        """Create a BathyEstimator object and set necessary informations

        :param image: the orthorectified image onto which bathymetry must be estimated.
        :param wave_params: parameters for the global and local bathymetry estimators
        :param nb_subtiles_max: Nb of subtiles for bathymetry estimation
        """
        # Store arguments in attributes for further use
        self.image = image
        self.waveparams = wave_params

        self._distoshore_provider: DisToShoreProvider = DefaultDisToShoreProvider()
        self._gravity_provider: GravityProvider = ConstantGravityProvider()

        # Create subtiles onto which bathymetry estimation will be done
        self.subtiles = SampledOrthoImage.build_subtiles(image, nb_subtiles_max,
                                                         self.waveparams.DXP, self.waveparams.DYP,
                                                         self.measure_extent)
        self._debug_samples: List[PointType] = []
        self._debug_sample = False

    @property
    @abstractproperty
    def bands_identifiers(self) -> List[str]:
        """ :returns: the spectral band identifiers in the product to use for bathymetry estimation
        """

    @property
    def smoothing_requested(self) -> bool:
        """ :returns: True if both smoothing columns and lines parameters are non zero
        """
        return self.smoothing_columns_size != 0 and self.smoothing_lines_size != 0

    @property
    def smoothing_columns_size(self) -> int:
        """ :returns: the size of the smoothing filter along columns in pixels
        """
        return self.waveparams.SM_LENGTH

    @property
    def smoothing_lines_size(self) -> int:
        """ :returns: the size of the smoothing filter along lines in pixels
        """
        return self.waveparams.SM_LENGTH

    @property
    def measure_extent(self) -> MarginsType:
        """ :returns: the cartographic extent to be used for bathy estimation around a point
        """
        return (self.waveparams.WINDOW / 2., self.waveparams.WINDOW / 2.,
                self.waveparams.WINDOW / 2., self.waveparams.WINDOW / 2.)

    # TODO: move get_phi_limits and get_kfft in spatial_dft_bathy_estimator
    def get_phi_limits(self, gravity: float,
                       wavenumbers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """  :returns: the minimum and maximum phase shifts for swallow and deep water at different
        wavenumbers

        :param gravity: the acceleration of the gravity to use (may vary locally)
        :param wavenumbers: the wavenumbers for which limits on phase are requested
        :returns: the requested sampling of the sinogram FFT
        """
        phi_min, phi_deep = phi_limits(wavenumbers, self.waveparams.DT,
                                       self.waveparams.MIN_D, gravity)

        return phi_deep, phi_min

    def get_kfft(self, gravity: float) -> np.ndarray:
        """  :returns: the requested sampling of the sinogram FFT

        :param gravity: the acceleration of the gravity to use (may vary locally)
        :returns: the requested sampling of the sinogram FFT
        """
        # frequencies based on wave characteristics:
        k_forced = 1 / ((np.arange(self.waveparams.MIN_T,
                                   self.waveparams.MAX_T,
                                   self.waveparams.STEP_T) ** 2) * gravity / (2. * np.pi))
        kfft = k_forced.reshape((k_forced.size, 1))

        return kfft

    @property
    def nb_subtiles(self) -> int:
        """ :returns: the number of subtiles
        """
        return len(self.subtiles)

    def compute_bathy(self, subtile_number: int) -> Dataset:
        """ Computes the bathymetry dataset for a given subtile.

        :param subtile_number: number of the subtile
        :returns: Subtile dataset
        """
        # Retrieve the subtile.
        subtile = self.subtiles[subtile_number]
        print(f'Subtile {subtile_number}: {self.image.short_name} {subtile}')

        # Build a bathymertry estimator over the subtile and launch estimation.
        subtile_estimator = OrthoBathyEstimator(self, subtile)
        dataset = subtile_estimator.compute_bathy()

        # Build the bathymetry dataset for the subtile.
        infos = subtile_estimator.build_infos()
        infos.update(self.image.build_infos())
        for key, value in infos.items():
            dataset.attrs[key] = value

        return dataset

# ++++++++++++++++++++++++++++ Debug support +++++++++++++++++++++++++++++

    def set_debug_samples(self, samples: List[PointType]) -> None:
        """ Sets the list of sample points to debug

        :param samples: a list of (X,Y) tuples defining the points to debug
        """
        self._debug_samples = samples

    def set_debug(self, sample: PointType) -> None:
        self._debug_sample = sample in self._debug_samples
        if self._debug_sample:
            print(f'Debugging point: X:{sample[0]} / Y:{sample[1]}')

    @property
    def debug_sample(self) -> bool:
        """ :returns: the current value of the debugging flag
        """
        return self._debug_sample

    def print_estimations_debug(self, waves_fields_estimations: WavesFieldsEstimations,
                                step: str) -> None:
        if self.debug_sample:
            print(f'estimations at step: {step}')
            for waves_field in waves_fields_estimations:
                print(waves_field)

# ++++++++++++++++++++++++++++ External data providers +++++++++++++++++++++++++++++

    def set_distoshore_provider(self, distoshore_provider: DisToShoreProvider) -> None:
        """ Sets the DisToShoreProvider to use with this estimator

        :param distoshore_provider: the DisToShoreProvider to use
        """
        self._distoshore_provider = distoshore_provider

    def get_distoshore(self, point: PointType) -> float:
        return self._distoshore_provider.get_distance(point)

    def set_gravity_provider(self, gravity_provider: GravityProvider) -> None:
        """ Sets the GravityProvider to use with this estimator

        :param gravity_provider: the GravityProvider to use
        """
        self._gravity_provider = gravity_provider

    def get_gravity(self, point: PointType, altitude: float=0.) -> float:
        return self._gravity_provider.get_gravity(point, altitude)
