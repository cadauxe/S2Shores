# -*- coding: utf-8 -*-
""" Definition of several functions useful for building an image tiling

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 05/05/2021

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
import math
from typing import Tuple


def modular_sampling(interval_min: float,
                     interval_max: float,
                     sampling_step: float) -> Tuple[int, int]:
    """ Given an interval of real numbers and a sampling step, select the start and stop samples
    inside the interval such that their values are multiple of the sampling step. Interval limits
    are considered as belonging to the interval.

    :param interval_min: the lower limit of the interval (assumed to belong to the interval)
    :param interval_max: the upper limit of the interval (assumed to belong to the interval)
    :param sampling_step: the sampling step of the interval
    :returns: start and stop sampling indexes such that:
              - interval_min <= start * sampling_step <= stop * sampling_step <= interval_max
    :raises ValueError: when interval is incorrectly specified or when no sample can be found in the
                        interval
    """
    if interval_min >= interval_max:
        msg = f'Cannot sample interval with incorrect limits: [{interval_min}, {interval_max}]'
        raise ValueError(msg)

    # Compute sample index and value at interval start
    start_sample_index = math.floor(interval_min / sampling_step)
    # If sample is strictly on the left of the interval, take the next sample on the right
    if start_sample_index * sampling_step < interval_min:
        start_sample_index += 1
    # Verify that the selected sample falls within the interval
    if start_sample_index * sampling_step > interval_max:
        msg = f'Start sample falls outside the interval: {start_sample_index * sampling_step}.'
        msg += ' Choose a lower sampling step or enlarge the interval width.'
        raise ValueError(msg)

    stop_sample_index = math.floor(interval_max / sampling_step)
    # If sample is strictly on the right of the interval, take the previous sample on the left
    if stop_sample_index * sampling_step > interval_max:
        stop_sample_index -= 1
    # Verify that the selected sample falls within the interval
    if stop_sample_index * sampling_step < interval_min:
        msg = f'Stop sample falls outside the interval: {stop_sample_index * sampling_step}.'
        msg += ' Choose a lower sampling step or enlarge the interval width.'
        raise ValueError(msg)

    return start_sample_index, stop_sample_index
