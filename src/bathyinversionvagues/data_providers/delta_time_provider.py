# -*- coding: utf-8 -*-
""" Definition of the DeltaTimeProvider abstract class and ConstantDeltaTimeProvider class

:author: GIROS Alain
:created: 02/08/2021
"""
from abc import abstractmethod
import datetime

from typing import Optional, Dict, Any, List

from ..image.image_geometry_types import PointType
from .localized_data_provider import LocalizedDataProvider


class DeltaTimeProvider(LocalizedDataProvider):
    """ A DeltaTimeProvider is a service able to provide the delta time at some position
    between two frames. The points where delta time is requested are specified by their coordinates
    in the image SRS.
    """

    @abstractmethod
    def get_delta_time(self, ref_frame_id: Any, sec_frame_id: Any,
                       point: Optional[PointType] = None) -> float:
        """ Provides the delta time at some point expressed by its X and Y coordinates in some SRS,
        between 2 frames specified by their ids. The frame id definition is left undefined and
        must be specified by subclasses.

        :param ref_frame_id: the id of the reference frame from which the duration will be counted
        :param sec_frame_id: the id of the secondary frame to which the duration will be counted
        :param point: a tuple containing the X and Y coordinates in the SRS set for this provider
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

    def get_delta_time(self, ref_frame_id: Any, sec_frame_id: Any,
                       point: Optional[PointType] = None) -> float:
        _ = point
        delta_time = self._frames_times[sec_frame_id] - self._frames_times[ref_frame_id]
        return delta_time.total_seconds()

    @property
    def chronology(self) -> List[Any]:
        """ :returns: a chronologically ordered list of the frames times keys.
        """
        return sorted(list(self._frames_times.keys()))
