# -*- coding: utf-8 -*-
"""
Class managing the computation of waves fields from two images taken at a small time interval.


:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 5 mars 2021
"""
from typing import Optional, Dict, Any, List  # @NoMove

from abc import abstractmethod, ABC

import numpy as np

from ..image_processing.waves_image import WavesImage
from .waves_field_estimation import WavesFieldEstimation


class LocalBathyEstimator(ABC):
    def __init__(self, images_sequence: List[WavesImage], global_estimator,
                 selected_directions: Optional[np.ndarray] = None) -> None:
        """ Constructor

        :param selected_directions: the set of directions onto which the sinogram must be computed
        """
        # TODO: Check that the images have the same resolution, satellite (and same size ?)
        self.global_estimator = global_estimator
        self._params = self.global_estimator.waveparams

        self.images_sequence = images_sequence
        self.selected_directions = selected_directions

        self.waves_fields_estimations: List[WavesFieldEstimation] = []

        self._metrics: Dict[str, Any] = {}

    @abstractmethod
    def run(self) -> None:
        """  Run the spatial bathymetry estimation, using some method specific to the inheriting
        class.

        This method return its results in waves_fields_estimations attribute as well as
        its metrics in _metrics attribute.
        """

    def get_filtered_results(self):
        # FIXME: Should this filtering be made at local estimation level ?
        filtered_out_waves_fields = [field for field in self.waves_fields_estimations if
                                     field.period >= self._params.MIN_T and
                                     field.period <= self._params.MAX_T]
        filtered_out_waves_fields = [field for field in filtered_out_waves_fields if
                                     field.ckg >= self._params.MIN_WAVES_LINEARITY and
                                     field.ckg <= self._params.MAX_WAVES_LINEARITY]
        # TODO: too high number of fields would provide a hint on poor quality measure
        print(len(filtered_out_waves_fields))
        return filtered_out_waves_fields

    def get_results_as_dict(self) -> Dict[str, np.ndarray]:
        """
        """
        filtered_out_waves_fields = self.get_filtered_results()
        nb_max_wave_fields = self._params.NKEEP

        delta_phase_ratios = np.empty((nb_max_wave_fields)) * np.nan
        celerities = np.empty((nb_max_wave_fields)) * np.nan  # Estimated celerity
        directions = np.empty((nb_max_wave_fields)) * np.nan
        deltas_phase = np.empty((nb_max_wave_fields)) * np.nan
        wavenumbers = np.empty((nb_max_wave_fields)) * np.nan
        wavelengths = np.empty((nb_max_wave_fields)) * np.nan
        energies_max = np.empty((nb_max_wave_fields)) * np.nan
        energies_ratios = np.empty((nb_max_wave_fields)) * np.nan
        depths = np.empty((nb_max_wave_fields)) * np.nan
        periods = np.empty((nb_max_wave_fields)) * np.nan
        periods_offshore = np.empty((nb_max_wave_fields)) * np.nan
        ckgs = np.empty((nb_max_wave_fields)) * np.nan
        delta_celerities = np.empty((nb_max_wave_fields)) * np.nan

        for ii, waves_field in enumerate(filtered_out_waves_fields[:nb_max_wave_fields]):
            directions[ii] = waves_field.direction
            wavenumbers[ii] = waves_field.wavenumber
            wavelengths[ii] = waves_field.wavelength
            celerities[ii] = waves_field.celerity
            periods[ii] = waves_field.period
            periods_offshore[ii] = waves_field.period_offshore
            ckgs[ii] = waves_field.ckg
            depths[ii] = waves_field.depth

            deltas_phase[ii] = waves_field.delta_phase
            delta_phase_ratios[ii] = waves_field.delta_phase_ratio
            energies_max[ii] = waves_field.energy_max
            energies_ratios[ii] = waves_field.energy_ratio

        return {'cel': celerities,
                'nu': wavenumbers,
                'L': wavelengths,
                'T': periods,
                'T_off': periods_offshore,
                'dir': directions,
                'dPhi': deltas_phase,
                'dPhiRat': delta_phase_ratios,
                'c2kg': ckgs,
                'energy': energies_max,
                'energyRat': energies_ratios,
                'depth': depths,
                'dcel': delta_celerities
                }

    @property
    def metrics(self) -> Dict[str, Any]:
        """ :returns: a dictionary of metrics concerning the estimation process
        """
        return self._metrics

    def _dump(self, variable: np.ndarray, variable_name: str) -> None:
        if variable is not None:
            print(f'{variable_name} {variable.shape} {variable.dtype}')
        print(variable)
