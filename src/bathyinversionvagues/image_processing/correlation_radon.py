# -*- coding: utf-8 -*-
"""
module -- Class encapsulating operations on the radon transform of a correlation matrix
"""

from .correlation_image import CorrelationImage
from .waves_radon import WavesRadon


class CorrelationRadon(WavesRadon):
    """
    This class extends class WavesRadon and is used for radon transform computed from a centered
    correlation matrix
    It is very important that circle=True has been chosen to compute radon matrix since we read
    values in meters from the axis of the sinogram
    """

    def __init__(self, image: CorrelationImage, directions_step: float = 1.,
                 weighted: bool = False) -> None:
        super().__init__(image, directions_step, weighted)
