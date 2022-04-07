# -*- coding: utf-8 -*-
"""
module -- Class encapsulating a list of superimposed images of the saize and resolution


:author: Alain Giros
:organization: CNES
:copyright: 2022 CNES. All rights reserved.
:license: see LICENSE file
:created: 7 april 2022
"""
from datetime import datetime

from typing import Tuple, Callable, List, Any, Union, Optional

import numpy as np

from ..image.image_geometry_types import ImageWindowType
from .waves_image import WavesImage


ImageProcessingFilters = List[Tuple[Callable, List[Any]]]
FrameIdType = Union[int, str, datetime]


class ImagesSequence(list):
    def __init__(self) -> None:
        """ Constructor
        """
        super().__init__()
        self._resolution: Optional[float] = None
        self._shape: Optional[Tuple[int, ...]] = None

        # FIXME: list or dict indexed by frame_id ???
        self._frames_id: List[FrameIdType] = []
        self._frames_time: List[datetime] = []

    @property
    def resolution(self) -> Optional[float]:
        """ :returns: The spatial resolution of this sequence of frames (m-1)"""
        return self._resolution

    @property
    def sampling_frequency(self) -> Optional[float]:
        """ :returns: The spatial sampling frequency of this sequence of frames (m-1)"""
        if self.resolution is None:
            return None
        return 1. / self.resolution

    def append_image(self, image: WavesImage, image_id: FrameIdType) -> None:
        if self._resolution is not None and image.resolution != self._resolution:
            msg = 'Trying to add an frame into images sequence with incompatible resolution:  new '
            msg += f'image resolution: {image.resolution} sequence resolution: {self.resolution}'
            raise ValueError(msg)
        if self._shape is not None and image.pixels.shape != self._shape:
            msg = 'Trying to add an image into frames sequence with incompatible shape:'
            msg += f' new image shape: {image.pixels.shape} sequence shape: {self._shape}'
            raise ValueError(msg)
        self._resolution = image.resolution
        self._shape = image.pixels.shape
        self.append(image)

        # TODO: make some checks on image_id
        self._frames_id.append(image_id)

    def extract_window(self, window: ImageWindowType) -> List[WavesImage]:

        # Create the sequence of WavesImages
        images_sequence: List[WavesImage] = []
        for sub_tile_image in self:
            window_image = sub_tile_image.extract_sub_image(window)
            images_sequence.append(window_image)
        return images_sequence
