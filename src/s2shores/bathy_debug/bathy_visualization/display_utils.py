# -*- coding: utf-8 -*-
"""
Utility functions to manage titles and some maths things.

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

from typing import TYPE_CHECKING  # @NoMove

import numpy as np

if TYPE_CHECKING:
    from ..local_bathymetry.spatial_correlation_bathy_estimator import (
        SpatialCorrelationBathyEstimator)  # @UnusedImport
    from ..local_bathymetry.spatial_dft_bathy_estimator import (
        SpatialDFTBathyEstimator)  # @UnusedImport


def get_display_title(local_estimator: 'SpatialDFTBathyEstimator') -> str:
    """ Get the title for the display of the estimator"""
    title = f'{local_estimator.global_estimator._ortho_stack.short_name} {local_estimator.location}'
    return title

def get_display_title_with_kernel(local_estimator: 'SpatialDFTBathyEstimator') -> str:
    """ Get the title for the display of the estimator with the kernel size"""
    title = f'{local_estimator.global_estimator._ortho_stack.short_name} {local_estimator.location}'
    smooth_kernel_xsize = local_estimator.global_estimator.smoothing_lines_size
    smooth_kernel_ysize = local_estimator.global_estimator.smoothing_columns_size
    filter_info = ''
    if smooth_kernel_xsize == 0 and smooth_kernel_ysize == 0:
        filter_info = f' (i.e. Smoothing Filter DEACTIVATED!)'

    return title + \
        f'\n Smoothing Kernel Size = [{2 * smooth_kernel_xsize + 1}px*{2 * smooth_kernel_ysize + 1}px]' + filter_info



def floor_to_nearest_10(val: float) -> float:
    """Round down to nearest 10."""
    return np.floor(val / 10.0) * 10.0

def ceil_to_nearest_10(val: float) -> float:
    """Round up to nearest 10."""
    return np.ceil(val / 10.0) * 10.0
