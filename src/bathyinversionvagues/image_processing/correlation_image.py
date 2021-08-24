# -*- coding: utf-8 -*-
"""
module -- Class encapsulating a correlation matrix onto which waves estimation will be made
"""
import numpy as np

from ..generic_utils.image_filters import funDetrend_2d, clipping
from .waves_image import WavesImage, ImageProcessingFilters


class CorrelationImage(WavesImage):
    def __init__(self, pixels: np.ndarray, resolution: float, tuning_ratio_size: float) -> None:
        super().__init__(pixels, resolution)

        preprocessing_filters: ImageProcessingFilters = []
        preprocessing_filters.append((funDetrend_2d, []))
        preprocessing_filters.append((clipping, [tuning_ratio_size]))
        self.apply_filters(preprocessing_filters)
