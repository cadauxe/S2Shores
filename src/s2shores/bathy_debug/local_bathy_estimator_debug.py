# -*- coding: utf-8 -*-
""" Base class for the estimators of wave fields from several images taken at a small
time interval.

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 5 mars 2021
"""
from abc import abstractmethod

from ..local_bathymetry.local_bathy_estimator import LocalBathyEstimator


class LocalBathyEstimatorDebug(LocalBathyEstimator):
    """ Abstract class handling debug mode for LocalBathyEstimator
    """

    def run(self) -> None:
        super().run()
        self.explore_results()

    @abstractmethod
    def explore_results(self) -> None:
        """ Method called when estimator has run to allow results exploration for debugging purposes
        """
