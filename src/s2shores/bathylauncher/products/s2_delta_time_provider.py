# -*- coding: utf-8 -*-
""" Definition of the S2DeltaTimeProvider class

:authors: see AUTHORS file
:organization : CNES, LEGOS, SHOM
:copyright: 2024 CNES. All rights reserved.
:created: 23 June 2021
:license: see LICENSE file


  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
import csv
from pathlib import Path

from typing import Dict, TYPE_CHECKING  # @NoMove

from shapely.geometry import Point


from s2shores.data_providers.delta_time_provider import (DeltaTimeProvider,
                                                         NoDeltaTimeValueError)


if TYPE_CHECKING:
    from .s2_image_product import S2ImageProduct  # @UnusedImport


class S2DeltaTimeProvider(DeltaTimeProvider):
    """ A DeltaTimeProvider which provides the delta time between the spectral bands of Sentinel 2
    at some position.
    """

    def __init__(self, delta_times_dir_path: Path, image: 'S2ImageProduct') -> None:
        """Create a DeltaTimeProvider using calibrated delta times between spectral bands of
        a specific S2 satellite (S2A or S2B)

        :param delta_times_dir_path: full path to the directory containing the CSV files giving
                                     the calibrated delta times of S2 satellites
        :param image: the S2 image for which delta times between bands will be requested.
        """
        super().__init__()
        self.s2_image = image
        delta_times_csv_filepath = delta_times_dir_path / f'{image.satellite}_delta_times.csv'
        self._delta_times = S2DeltaTimeProvider.read_delta_times(delta_times_csv_filepath)

    def get_delta_time(self, first_frame_id: str, second_frame_id: str, point: Point) -> float:
        """ Provides the delta time at some point between 2 spectral bands of an S2 product.

        :param first_frame_id: the reference spectral band id
        :param second_frame_id: the secondary spectral band id
        :param point: a point expressed in the SRS coordinates set for this provider
        :returns: the acquisition time difference between 2 spectral bands at the requested position
        :raises NoDeltaTimeValueError: when the point is outside all detectors footprints
        """
        first_det_index = self.s2_image.find_detector(first_frame_id, point)
        # FIXME: it is surprising that the detectors for different spectral bands are supposed to
        # be exactly superimposable, thus making detector finding in second band useless.
        # second_det_index = self.s2_image.find_detector(second_frame_id, point)

        # In the S2 convention, sign(delta time) is coherent with the ground velocity
        # of the satellite.
        # For instance, if S2deltaT(B2,B4,Di) is positive, then B4 is the first band
        # that acquires the point and B2 is the second.
        if first_det_index is None:
            raise NoDeltaTimeValueError()
        delta_t, hsat, vground = self._delta_times[first_frame_id][second_frame_id][first_det_index]
        return delta_t

    @staticmethod
    def read_delta_times(delta_times_csv_filepath: Path) -> Dict[str, dict]:
        """ Read the CSV file containing the calibrated time differences between spectral bands
        of Sentinel2 sensors.

        :param delta_times_csv_filepath: the full path to a CSV file describing the time differences
                                         of all the possible pairs of detectors and spectral bands
                                         for a given S2 sensor.
        :returns: a two level dictionary of dictionaries indexed firstly by the reference spectral
                  band id and secondly by the secondary spectral band id. Each entry then contain
                  a fixed size list of delta times between detectors with the same index in both
                  spectral bands.
        """
        from .s2_image_product import NB_BANDS_DETECTORS
        delta_times: Dict[str, dict] = {}
        with open(delta_times_csv_filepath, newline='') as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
            csvfile.seek(0)  # Reset file pointer after reading for sniffing

            reader = csv.DictReader(csvfile, delimiter=dialect.delimiter)
            for row in reader:
                src_band_id = row['bande_src']
                if src_band_id not in delta_times.keys():
                    delta_times[src_band_id] = {}

                dst_band_id = row['bande_dst']
                if dst_band_id not in delta_times[src_band_id].keys():
                    delta_times[src_band_id][dst_band_id] = [0.] * NB_BANDS_DETECTORS

                det_index = int(row['detecteur'][1:]) - 1

                delta_t = float(row['delta_t'])
                hsat_str = row.get('Hsat')
                hsat = None if hsat_str is None else float(hsat_str)
                vground_str = row.get('vground')
                vground = None if vground_str is None else float(vground_str)
                delta_times[src_band_id][dst_band_id][det_index] = (delta_t, hsat, vground)
        return delta_times
