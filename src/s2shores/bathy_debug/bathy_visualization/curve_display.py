# -*- coding: utf-8 -*-
"""
Functions for managing axes.

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

import numpy as np
import matplotlib.pyplot as plt

def display_curve(data: np.ndarray, legend: str) -> None:
    """Display a single curve with legend."""
    _, ax = plt.subplots()
    ax.plot(data)
    ax.set_title(legend)

def display_3curves(data1: np.ndarray, data2: np.ndarray, data3: np.ndarray) -> None:
    """Display three curves in separate subplots."""
    _, ax = plt.subplots(3)
    ax[0].plot(data1)
    ax[1].plot(data2)
    ax[2].plot(data3)

def display_4curves(data1: np.ndarray, data2: np.ndarray,
                   data3: np.ndarray, data4: np.ndarray) -> None:
    """Display four curves in a 2x2 grid."""
    _, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].plot(data1)
    ax[1, 0].plot(data2)
    ax[0, 1].plot(data3)
    ax[1, 1].plot(data4)

def display_image(data: np.ndarray, legend: str) -> None:
    """Display an image with legend."""
    _, ax = plt.subplots()
    ax.imshow(data, aspect='auto', cmap='gray')
    ax.set_title(legend)