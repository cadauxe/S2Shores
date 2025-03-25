# -*- coding: utf-8 -*-
""" Definition of the class and constants related to GeoTIFF products handling

:authors: see AUTHORS file
:organization : CNES, LEGOS, SHOM
:copyright: 2024 CNES. All rights reserved.
:created: 3 December 2021
:license: see LICENSE file


  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Optional  # @NoMove


from s2shores.data_providers.delta_time_provider import (ConstantDeltaTimeProvider,
                                                         FramesTimesDict)
from s2shores.image.ortho_stack import OrthoStack, FrameIdType, FramesIdsType


class GeoTiffProduct(OrthoStack):
    """ Class holding attributes and methods related to a geotiff product.
    """

    def __init__(self, product_path: Path) -> None:
        """ Constructor.

        :param product_path: path of geotiff product
        """
        metadata_file_name = product_path.with_suffix('.json')
        with open(metadata_file_name) as metadata_file:
            self._metadata_dict = json.load(metadata_file)
        super().__init__(product_path)

    @property
    def short_name(self) -> str:
        """ :returns: a short name for this product
        """
        return f'{self.zone_id}_{self.acquisition_time}'

    @property
    def full_name(self) -> str:
        """ :returns: a full name for this product
        """
        return self.short_name

    @property
    def zone_id(self) -> str:
        """ :returns: the id of the zone corresponding to this product
        """
        return self._metadata_dict['ZONE_ID']

    @property
    def acquisition_time(self) -> str:
        """ :returns: the acquisition time of this product
        """
        return self._metadata_dict['ACQUISITION_TIME']

    @property
    def satellite(self) -> str:
        """ :returns: the satellite that acquired this product
        """
        return self._metadata_dict['SATELLITE']

    @property
    def preprocessing_level(self) -> str:
        """ :returns: the preprocessing level of this product
        """
        return self._metadata_dict['PROCESSING_LEVEL']

    @property
    def frames_time(self) -> FramesTimesDict:
        """ :returns: the times of each frames contained in this product
        """
        frames_times = {}
        for key, value in self._metadata_dict['FRAMES_TIME'].items():
            frames_times[int(key)] = datetime.fromisoformat(value)
        return frames_times

    @property
    def usable_frames(self) -> FramesIdsType:
        """ :returns: the list of frames contained in this product
        """
        return sorted(list(self.frames_time.keys()))

    def get_image_file_path(self, frame_id: FrameIdType) -> Path:
        """ :returns: the path of the image file corresponding to the given frame id
        """
        _ = frame_id
        # All frames are contained in a single file
        return self.product_path

    def get_frame_index_in_file(self, frame_id: FrameIdType) -> int:
        """ :returns: the index of the frame in the image file corresponding to the given frame id
        """
        try:
            frame_id_int = int(frame_id)  # type: ignore
        except ValueError as excp:
            raise TypeError('GeoTiffProduct frame index cannot be converted to int') from excp
        return frame_id_int

    def build_infos(self) -> Dict[str, str]:
        """ :returns: a dictionary of metadata describing this geotiff product
        """
        infos = {
            'idZone': self.zone_id,
            'Proc_level': self.preprocessing_level,
            'Proc_baseline': 'NONE',
            'Rel_Orbit': 'NONE'}

        # metadata from the base class
        infos.update(super().build_infos())
        return infos

    def create_delta_time_provider(
            self, external_delta_times_path: Optional[Path] = None) -> ConstantDeltaTimeProvider:
        """ :returns: a delta time provider for this product
        """
        _ = external_delta_times_path

        return ConstantDeltaTimeProvider(self.frames_time)
