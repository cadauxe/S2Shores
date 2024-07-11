# -*- coding: utf-8 -*-
"""
Module to display a pseudo RGB image with a polar plot in the foreground

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2024 CNES. All rights reserved.
:created: 4 November 2024
:license: see LICENSE file


  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""

from typing import List, Optional, Tuple  # @NoMove


from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from .polar_display import build_polar_plot


def create_pseudorgb(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """Create a pseudo-RGB image from two input images."""
    normalized_im2 = (image2 - image2.min()) / (image2.max() - image2.min())
    normalized_im1 = (image1 - image1.min()) / (image1.max() - image1.min())

    ps_rgb = np.dstack((normalized_im2, normalized_im1, normalized_im2))
    ps_rgb = ps_rgb - ps_rgb.min()
    return ps_rgb / (ps_rgb.max() - ps_rgb.min())


def build_display_pseudorgb(fig: Figure, axes: Axes, title: str, image: np.ndarray,
                            resolution: float,
                            subplot_pos: [float, float, float],
                            directions: Optional[List[Tuple[float, float]]] = None,
                            cmap: Optional[str] = None, coordinates: bool=True) -> None:

    build_polar_plot(fig, axes, image, resolution, subplot_pos, directions, cmap,
                     coordinates, polar_labels=['0°', '90°', '180°', '-90°'])
