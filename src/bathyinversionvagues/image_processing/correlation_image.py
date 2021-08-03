# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import numpy as np

from .waves_image import WavesImage
from .shoresutils import funDetrend_2d


class CorrelationImage(WavesImage):
    def __init__(self, pixels: np.ndarray, resolution: float, tuning_ratio_size: float,
                 detrend: bool = True, smoothing: Optional[Tuple[int, int]] = None) -> None:
        self.tuning_ratio_size = tuning_ratio_size
        super().__init__(pixels, resolution, detrend, smoothing)

    def detrend(self):
        self.pixels = funDetrend_2d(self.pixels)
        s1, s2 = np.shape(self.pixels)
        self.pixels = self.pixels[int(s1 / 2 - self.tuning_ratio_size * s1 / 2):int(
            s1 / 2 + self.tuning_ratio_size * s1 / 2),
                      int(s2 / 2 - self.tuning_ratio_size * s2 / 2):int(
                          s2 / 2 + self.tuning_ratio_size * s2 / 2)]
