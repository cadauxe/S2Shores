# -*- coding: utf-8 -*-
""" Class encapsulating a list of superimposed images of the same size and resolution


:author: Alain Giros
:organization: CNES
:copyright: 2022 CNES. All rights reserved.
:license: see LICENSE file
:created: 7 april 2022
"""
from datetime import datetime

from typing import Tuple, Callable, List, Any, Union, Optional

from ..image.image_geometry_types import ImageWindowType
from .waves_image import WavesImage


ImageProcessingFilters = List[Tuple[Callable, List[Any]]]
FrameIdType = Union[str, int, datetime]
FramesIdsType = Union[List[str], List[int], List[datetime]]


# FIXME: list or dict indexed by image_id ???
class ImagesSequence(list):
    """ Class encapsulating the information describing a sequence of superimposed images of same
    shape and resolution and providing operations on it.
    """

    def __init__(self) -> None:
        super().__init__()
        self._resolution = 0.
        self._shape: Optional[Tuple[int, ...]] = None

        self._images_id: List[FrameIdType] = []
        self._images_time: List[datetime] = []

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """ :returns: The shape common to all the images contained in this sequence of images"""
        return self._shape

    @property
    def resolution(self) -> float:
        """ :returns: The spatial resolution of this sequence of images (m)"""
        return self._resolution

    @property
    def sampling_frequency(self) -> Optional[float]:
        """ :returns: The spatial sampling frequency of this sequence of images (m-1)"""
        if self.resolution is None:
            return None
        return 1. / self.resolution

    def append_image(self, image: WavesImage, image_id: FrameIdType) -> None:
        """ Append a new image to this image sequence. The first appended image fixes the spatial
        resolution and the shape of all the image which will be entered in the sequence.

        :param image: the image to append at the last element of the sequence
        :param image_id: the identifier of the image, must be unique in the sequence.
        :raises ValueError: when the image has not the same shape or resolution than the images
                            already recorded or when the image identifier is already present in the
                            sequence.
        """
        if self._resolution != 0. and image.resolution != self._resolution:
            msg = 'Trying to add an image into images sequence with incompatible resolution:  new '
            msg += f'image resolution: {image.resolution} sequence resolution: {self.resolution}'
            raise ValueError(msg)
        if self._shape is not None and image.pixels.shape != self._shape:
            msg = 'Trying to add an image into images sequence with incompatible shape:'
            msg += f' new image shape: {image.pixels.shape} sequence shape: {self._shape}'
            raise ValueError(msg)
        if image_id in self._images_id:
            msg = 'Trying to add an image into images sequence with an already existing identifier:'
            msg += f' {image_id}'
            raise ValueError(msg)
        self._resolution = image.resolution
        self._shape = image.pixels.shape
        self.append(image)
        self._images_id.append(image_id)

    def extract_window(self, window: ImageWindowType) -> 'ImagesSequence':
        """ Extract a new images sequence by taking pixels from a window contained within the
        sequence shape.

        :param window: a window defined within the shape of this images sequence:
                       (line_start, line_stop, column_start, column_stop)
        :returns: an images sequence built with the excerpts extracted from the images over the
                  window. It has the same resolution and number of images as this sequence
                  and the image identifiers are copied from the image identifiers of this sequence.
        """
        images_sequence = ImagesSequence()
        for index, sub_tile_image in enumerate(self):
            window_image = sub_tile_image.extract_sub_image(window)
            images_sequence.append_image(window_image, self._images_id[index])
        return images_sequence
