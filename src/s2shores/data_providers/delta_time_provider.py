# -*- coding: utf-8 -*-
""" Definition of the DeltaTimeProvider abstract class and ConstantDeltaTimeProvider class

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 02/08/2021

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, List  # @NoMove

from shapely.geometry import Point

from ..waves_exceptions import WavesException
from .localized_data_provider import LocalizedDataProvider


class NoDeltaTimeValueError(WavesException):
    """ Exception raised when a DeltaTimeProvider cannot provide a delta time at some point
    """


class NoDeltaTimeProviderError(WavesException):
    """ Exception raised when using bathymetry estimator without specifying a DeltaTimeProvider
    """


class DeltaTimeProvider(ABC, LocalizedDataProvider):
    """ A DeltaTimeProvider is a service able to provide the delta time at some position
    between two frames. The points where delta time is requested are specified by their coordinates
    in the image SRS.
    """

    @abstractmethod
    def get_delta_time(self, first_frame_id: Any, second_frame_id: Any, point: Point) -> float:
        """ Provides the delta time at some point expressed by its X and Y coordinates in some SRS,
        between 2 frames specified by their ids. The frame id definition is left undefined and
        must be specified by subclasses.

        :param first_frame_id: the id of the frame from which the duration will be counted
        :param second_frame_id: the id of the frame to which the duration will be counted
        :param point: a point expressed in the SRS coordinates set for this provider
        :returns: the delta time between frames at this point (s).
        """


# Type allowing to describe the acquisition date and time of the different frames submitted to
# the ConstantDeltaTimeProvider.
FramesTimesDict = Dict[Any, datetime.datetime]


class ConstantDeltaTimeProvider(DeltaTimeProvider):
    """ A DeltaTimeProvider which provides a constant delta time between any 2 frames whatever
    the requested position.
    """

    def __init__(self, frames_times: FramesTimesDict) -> None:
        """ Constructor

        :param frames_times: a dictionary providing the acquisition date and time of the different
                             frames. Dictionary keys are any kind of frame identifier which will be
                             used for identifying those frames when calling get_delta_time().
        """
        super().__init__()
        self._frames_times = frames_times

    def get_delta_time(self, first_frame_id: Any, second_frame_id: Any, point: Point) -> float:
        _ = point
        delta_time = self._frames_times[second_frame_id] - self._frames_times[first_frame_id]
        return delta_time.total_seconds()

    @property
    def chronology(self) -> List[Any]:
        """ :returns: a chronologically ordered list of the frames times keys.
        """
        return sorted(list(self._frames_times.keys()))
