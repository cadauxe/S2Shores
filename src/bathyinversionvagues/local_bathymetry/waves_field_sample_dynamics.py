# -*- coding: utf-8 -*-
""" Class handling the information describing a waves field sample..

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 6 mars 2021
"""
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
        self.register_wavelength_change(self.ensure_physical_consistency)

    def ensure_physical_consistency(self) -> None:
        if np.isnan(self.wavelength):
            self._period = np.nan
            self._celerity = np.nan
        else:
            if not np.isnan(self.celerity):
                if np.isnan(self.period):
                    self._period = self.wavelength / self.celerity
                else:
                    self._period = np.nan
                    self._celerity = np.nan
            else:
                if not np.isnan(self.period):
                    self._celerity = self.wavelength / self.period

    @property
    def period(self) -> float:
        """ :returns: The waves field period (s) """
        return self._period

    @period.setter
    def period(self, value: float) -> None:
        self._period = value
        self.ensure_physical_consistency()

    @property
    def celerity(self) -> float:
        """ :returns: The waves field velocity (m/s) either the celerity which was directly set or
                      computed from the wavelength and the period
        """
        return self._celerity

    @celerity.setter
    def celerity(self, value: float) -> None:
        self._celerity = value
        self.ensure_physical_consistency()

    def __str__(self) -> str:
        result = WavesFieldSampleGeometry.__str__(self)
        result += f'\nperiod: {self.period:5.2f} (s)  celerity: {self.celerity:5.2f}'
        return result
