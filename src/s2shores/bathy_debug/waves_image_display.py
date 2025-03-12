# -*- coding: utf-8 -*-
"""
Class managing the displau of waves images.


:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 5 mars 2021

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
import os
from typing import List, Optional, Tuple  # @NoMove

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

def create_pseudorgb(image1: np.ndarray, image2: np.ndarray,) -> np.ndarray:
    normalized_im2 = (image2 - image2.min()) / (image2.max() - image2.min())
    normalized_im1 = (image1 - image1.min()) / (image1.max() - image1.min())

    ps_rgb = np.dstack((normalized_im2, normalized_im1, normalized_im2))
    ps_rgb = ps_rgb - ps_rgb.min()
    return ps_rgb / (ps_rgb.max() - ps_rgb.min()) 


def build_display_waves_image(fig: Figure, axes: Axes, title: str, image: np.ndarray,
                              resolution: float,
                              subplot_pos: [float, float, float],
                              directions: Optional[List[Tuple[float, float]]] = None,
                              cmap: Optional[str] = None, coordinates: bool=True) -> None:

    (l1, l2) = np.shape(image)
    imin = np.min(image)
    imax = np.max(image)
    axes.imshow(image, norm=Normalize(vmin=imin, vmax=imax), cmap=cmap)
    # create polar axes in the foreground and remove its background to see through
    subplot_locator = int(f'{subplot_pos[0]}{subplot_pos[1]}{subplot_pos[2]}')
    ax_polar = fig.add_subplot(subplot_locator, polar=True)
    ax_polar.set_yticklabels([])
    polar_ticks = np.arange(4) * np.pi / 2.
    polar_labels = ['0°', '90°', '180°', '-90°']

    plt.xticks(polar_ticks, polar_labels, size=9, color='blue')
    for i, label in enumerate(ax_polar.get_xticklabels()):
        label.set_rotation(i * 45)
    ax_polar.set_facecolor('None')

    xmax = f'{l1}px \n {np.round((l1-1)*resolution)}m'
    axes.set_xticks([0, l1 - 1], ['0', xmax], fontsize=8)
    ymax = f'{l2}px \n {np.round((l2-1)*resolution)}m'
    if coordinates:
        axes.set_yticks([0, l2 - 1], ['0', ymax], fontsize=8)
    else:
        axes.set_yticks([0, l2 - 1], ['', ''], fontsize=8)
        axes.set_xticks([0, l1 - 1], ['\n', ' \n'], fontsize=8)

    if directions is not None:
        # Normalization of arrows length
        coeff_length_max = np.max((list(zip(*directions))[1]))
        radius = np.floor(min(l1, l2) / 2) - 5
        for direction, coeff_length in directions:
            arrow_length = radius * coeff_length / coeff_length_max
            dir_rad = np.deg2rad(direction)
            axes.arrow(l1 // 2, l2 // 2,
                       np.cos(dir_rad) * arrow_length, -np.sin(dir_rad) * arrow_length,
                       head_width=2, head_length=3, color='r')

    axes.xaxis.tick_top()
    axes.set_title(title, fontsize=9, loc='center')
    # Manage blank spaces
    # plt.tight_layout()
    
def build_display_pseudorgb(fig: Figure, axes: Axes, title: str, image: np.ndarray,
                            resolution: float,
                            subplot_pos: [float, float, float],
                            directions: Optional[List[Tuple[float, float]]] = None,
                            cmap: Optional[str] = None, coordinates: bool=True) -> None:

    (l1, l2, l3) = np.shape(image)
    imin = np.min(image)
    imax = np.max(image)
    axes.imshow(image, norm=Normalize(vmin=imin, vmax=imax), cmap=cmap)
    # create polar axes in the foreground and remove its background to see through
    subplot_locator = int(f'{subplot_pos[0]}{subplot_pos[1]}{subplot_pos[2]}')
    ax_polar = fig.add_subplot(subplot_locator, polar=True)
    ax_polar.set_yticklabels([])
    polar_ticks = np.arange(4) * np.pi / 2.
    polar_labels = ['0°', '90°', '180°', '-90°']

    plt.xticks(polar_ticks, polar_labels, size=9, color='blue')
    for i, label in enumerate(ax_polar.get_xticklabels()):
        label.set_rotation(i * 45)
    ax_polar.set_facecolor('None')

    xmax = f'{l1}px \n {np.round((l1-1)*resolution)}m'
    axes.set_xticks([0, l1 - 1], ['0', xmax], fontsize=8)
    ymax = f'{l2}px \n {np.round((l2-1)*resolution)}m'

    if coordinates:
        axes.set_yticks([0, l2 - 1], [ymax, '0'], fontsize=8)
    else:
        axes.set_yticks([0, l2 - 1], ['', ''], fontsize=8)
        axes.set_xticks([0, l1 - 1], ['\n', ' \n'], fontsize=8)

    if directions is not None:
        # Normalization of arrows length
        coeff_length_max = np.max((list(zip(*directions))[1]))
        radius = np.floor(min(l1, l2) / 2) - 5
        for direction, coeff_length in directions:
            arrow_length = radius * coeff_length / coeff_length_max
            dir_rad = np.deg2rad(direction) + np.pi
            axes.arrow(l1 // 2, l2 // 2,
                       np.cos(dir_rad) * arrow_length, -np.sin(dir_rad) * arrow_length,
                       head_width=2, head_length=3, color='r')

    axes.xaxis.tick_top()
    axes.set_title(title, fontsize=9, loc='center')

