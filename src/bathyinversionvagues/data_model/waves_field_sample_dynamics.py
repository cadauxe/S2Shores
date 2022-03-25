# -*- coding: utf-8 -*-
""" Class handling the information describing a waves field sample..

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 6 mars 2021
"""
from typing import List, Callable

import numpy as np

from .waves_field_sample_geometry import WavesFieldSampleGeometry


class WavesFieldSampleDynamics(WavesFieldSampleGeometry):
    """ This class encapsulates the information related to the dynamics of a waves field sample.
    It inherits from WavesFieldSampleGeometry which describes the observed field geometry,
    and contains specific attributes related to the field dynamics:

    - its period
    - its celerity

    """

    def __init__(self) -> None:
        super().__init__()
        self._period = np.nan
        self._celerity = np.nan
        self._period_change_observers: List[Callable] = []

        self.register_wavelength_change(self.wavelength_change_in_dynamics)

    @property
    def period(self) -> float:
        """ :returns: The waves field period (s), which was either externally provided or computed
                      from the wavelength and the celerity
        :raises ValueError: when the period is not positive.
        """
        return self._period

    @period.setter
    def period(self, value: float) -> None:
        if value != self._period:
            if value < 0.:
                raise ValueError('Period must be positive')
            self._period = value
            if not np.isnan(self.celerity) and not np.isnan(self.wavelength):
                self._celerity = np.nan
                self.wavelength = np.nan
            self._solve_movement_equation()
            for notify in self._period_change_observers:
                notify()

    def is_period_valid(self, period_min: float, period_max: float) -> bool:
        """ Check if the waves field period is valid.

        :param period_min: minimum value allowed for the period
        :param period_max: maximum value allowed for the period
        :returns: True if the period is between the minimum and maximum values, False otherwise
        """
        return self.period >= period_min and self.period <= period_max

    @property
    def celerity(self) -> float:
        """ :returns: The waves field velocity (m/s), which was either externally provided or
                      computed from the wavelength and the period
        :raises ValueError: when the celerity is not positive.
        """
        return self._celerity

    @celerity.setter
    def celerity(self, value: float) -> None:
        if value != self.celerity:
            if value < 0:
                raise ValueError('Celerity must be positive')
            self._celerity = value
            if not np.isnan(self.period) and not np.isnan(self.wavelength):
                self._period = np.nan
                self.wavelength = np.nan
            self._solve_movement_equation()

    def register_period_change(self, notify: Callable) -> None:
        """ Register the functions to be called whenever a change of the period value occurs.

        :param notify: a function without argument which must be called when the period value
                       is changed
        """
        self._period_change_observers.append(notify)

    def wavelength_change_in_dynamics(self) -> None:
        """ When wavelength has changed (new value is ensured to be different from the previous one)
        either reset period and celerity if both were set, or update one of them if the other is set
        """
        if not np.isnan(self.period) and not np.isnan(self.celerity):
            self._period = np.nan
            self._celerity = np.nan
        self._solve_movement_equation()

    def _solve_movement_equation(self) -> None:
        """ Solves the movement equation ( L=c*T ) when exactly one of the 3 variables is not set.
        In other cases does not change anything.
        """
        wavelength_set = not np.isnan(self.wavelength)
        period_set = not np.isnan(self.period)
        celerity_set = not np.isnan(self.celerity)
        if wavelength_set and period_set and not celerity_set:
            self._celerity = self.wavelength / self.period
        elif wavelength_set and not period_set and celerity_set:
            self._period = self.wavelength / self.celerity
        elif not wavelength_set and period_set and celerity_set:
            self.wavelength = self.celerity * self.period

    def __str__(self) -> str:
        result = WavesFieldSampleGeometry.__str__(self)
        result += f'\nDynamics:   period: {self.period:5.2f} (s)  '
        result += f'celerity: {self.celerity:5.2f} (m/s)'
        return result
