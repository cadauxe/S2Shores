# -*- coding: utf-8 -*-
""" Definition of the class and constants related to S2 products handling

:authors: see AUTHORS file
:organization : CNES, LEGOS, SHOM
:copyright: 2024 CNES. All rights reserved.
:created: 11 May 2021
:license: see LICENSE file


  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # @NoMove

import xml.etree.ElementTree as ET
import numpy as np  # @NoMove

from shapely.geometry import Polygon, Point
from shapely.geometry.linestring import LineString
from osgeo import gdal, ogr

from s2shores.image.ortho_stack import OrthoStack, FrameIdType, FramesIdsType

from .s2_delta_time_provider import S2DeltaTimeProvider


S2_SPECTRAL_BANDS_NAMES = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06',
                           'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']

S2_SPECTRAL_BANDS_10M = ['B02', 'B03', 'B04', 'B08']
S2_SPECTRAL_BANDS_20M = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
S2_SPECTRAL_BANDS_60M = ['B01', 'B09', 'B10']

NB_BANDS_DETECTORS = 12

S2_TILE_SIZE = 109800 # Sentinel 2 tile size in meters (X & Y)

class S2ImageProduct(OrthoStack):
    """ Class holding attributes and methods related to a sentinel2 product dataset.
    """

    def __init__(self, product_path: Path) -> None:
        """ Constructor.

        :param product_path: Path to the directory corresponding to the S2 product
        """
        super().__init__(product_path)

        # Dictionary to store the detectors footprints defined for this S2 product
        self._bands_detectors: Optional[Dict[FrameIdType, List[Polygon]]] = None
        self._sun_angle_grids: Optional[Dict[FrameIdType, List[Polygon]]] = None
        self._viewing_angle_grids: Optional[Dict[FrameIdType, List[Polygon]]] = None

    @property
    def full_name(self) -> str:
        """ :returns: the full name of this product from which metadata will be parsed
        """
        return self.product_path.stem

    @property
    def short_name(self) -> str:
        """ :returns: a short name for this product
        """
        return f'{self.zone_id}_{self.acquisition_time}'

    @property
    def satellite(self) -> str:
        """ :returns: the satellite that acquired this product
        """
        metadata = self.full_name.split('_')
        return metadata[0]

    @property
    def zone_id(self) -> str:
        """ :returns: the tile id of this S2 image
        """
        metadata = self.full_name.split('_')
        return metadata[5]

    @property
    def acquisition_time(self) -> str:
        """ :returns: the acquisition time of this product
        """
        metadata = self.full_name.split('_')
        return metadata[2]

    @property
    def processing_baseline(self) -> int:
        """ :returns: the processing baseline of this product
        """
        metadata = self.full_name.split('_')
        return int(metadata[3][1:])

    @property
    def orbit_number(self) -> int:
        """ :returns: the orbit number of this product
        """
        metadata = self.full_name.split('_')
        return int(metadata[4][1:])

    @property
    def s2_product_path(self) -> Path:
        """ Returns the path to the first S2 product in the granule directory

        :raises FileNotFoundError: if the GRANULE directory is not found or if it does not contain
                                   a subdirectory.
        """
        # path to the images
        granule_path = self.product_path / 'GRANULE'
        if not granule_path.is_dir():
            raise FileNotFoundError(f'S2 granule not found: {granule_path}')
        for images_path in granule_path.iterdir():
            # break the loop for the first retrieved subdirectory in the granule
            if images_path.is_dir():
                return images_path
        msg = f'Unable to find a subdirectory inside the S2 granule: {granule_path}'
        raise FileNotFoundError(msg)

    @property
    def s2_metadata_path(self) -> Path:
        """ :returns: the path to the XML file containing the metadata of an S2 product
        """
        return self.s2_product_path / 'MTD_TL.xml'

    @property
    def s2_product_metadata_path(self) -> Path:
        """ :returns: the path to the XML file containing the metadata of an S2 product
        """
        return self.product_path / 'MTD_MSIL1C.xml'

    # TODO: Find a way to select the spectral bands to use according to the resolution
    @property
    def usable_frames(self) -> FramesIdsType:
        """ :returns: the list of frames contained in this product
        """
        return S2_SPECTRAL_BANDS_10M

    def get_image_file_path(self, frame_id: FrameIdType) -> Path:
        """ :returns: the path of the image file corresponding to the given frame id
        """
        if not isinstance(frame_id, str):
            raise TypeError('S2ImageProduct only supports bands names for indexing frames')
        return self.s2_product_path / 'IMG_DATA' / f'{self.short_name}_{frame_id}.jp2'

    def get_frame_index_in_file(self, frame_id: FrameIdType) -> int:
        """ :returns: the index of the frame in the image file
        """
        _ = frame_id
        # There is only a single band in one S2 image file
        return 1

    def get_viewing_angles_grids(self, working_bands: List[FrameIdType]) -> Dict[FrameIdType,
                                                                                 np.ndarray]:
        """
        :return: Viewing Incidence Angles Grids for each detectors of each bands
                 and compute the mean angle over detectors for each band.
        """
        tree_node = ET.parse(self.s2_metadata_path).getroot()
        node_list = tree_node.findall('.//Viewing_Incidence_Angles_Grids')

        angles_tag = []
        angles = {}
        for node in node_list:
            dect_id = int(node.attrib['detectorId'])
            band_id = f'B{int(node.attrib["bandId"]):02}'
            if band_id=='B00':
                band_id = 'B8A' # Band 8A id is 00, swap it for clearer understanding
                #TODO : SURE ??? TO CHECK, cf DATASTRIP.xml
            if band_id in working_bands:
                if not band_id in angles:
                    angles[band_id] = {}
                if not dect_id in angles[band_id]:
                    angles[band_id][dect_id] = {}

                for angle in node:
                    if not angle.tag in angles_tag:
                        angles_tag.append(angle.tag)
                    values = angle.findall('.//VALUES')
                    angle_arr = np.array(
                        list(map(lambda x: x.text.split(' '), values))).astype('float')
                    angles[band_id][dect_id][angle.tag] = angle_arr

        for band_id in angles.keys():
            for angle_tag in angles_tag:
                angles[band_id][angle_tag] = np.nanmean(
                    np.dstack([angles[band_id][dect_id][angle_tag]
                               for dect_id in angles[band_id] if not dect_id in angles_tag]),
                    axis=2)

        return angles

    def get_sun_angles_grid(self) -> Dict[FrameIdType, np.ndarray]:
        """
        :return: Sun Angles Grids for each detectors of each bands
                 and compute the mean angle over detectors for each band.
        """
        tree_node = ET.parse(self.s2_metadata_path).getroot()
        node_list = tree_node.findall('.//Sun_Angles_Grid/')

        angles_tag = []
        angles = {}
        for angle in node_list:
            if not angle.tag in angles_tag:
                angles_tag.append(angle.tag)
            values = angle.findall('.//VALUES')
            angle_arr = np.array(list(map(lambda x: x.text.split(' '), values))).astype('float')
            angles[angle.tag] = angle_arr

        return angles

    def build_infos(self) -> Dict[str, str]:
        """ :returns: a dictionary of metadata describing this S2 image
        """
        metadata = self.full_name.split('_')

        infos = {
            'idZone': self.zone_id,  #
            'Proc_level': metadata[1],  #
            'Proc_baseline': metadata[3],  #
            'Rel_Orbit': metadata[4]}  #

        # metadata from the base class
        infos.update(super().build_infos())
        return infos

    def find_detector(self, band_id: str, point: Point) -> Optional[int]:
        """ Find the detector in one spectral band which has acquired the specified point.

        :param band_id: the id of the spectral band where to search for the detector.
        :param point: the point expressed in the S2 product SRS coordinates
        :returns: the index of the detector in the spectral band which has acquired the specified
                  point, or None if the point lies outside the detectors footprints of the S2 tile.
        """
        band_detectors = self.bands_detectors[band_id]
        found_detectors = []
        for detector_index, detector_footprint in enumerate(band_detectors):
            if detector_footprint is None:
                continue
            if detector_footprint.contains(point):
                # FIXME: Perf improvement: test if next detector contains the point, then break.
                found_detectors.append((detector_index, detector_footprint))

        if not found_detectors:
            return None
        if len(found_detectors) > 1:
            found_detector = self._choose_overlapping_detectors(found_detectors, point)
        else:
            found_detector = found_detectors[0][0]
        return found_detector

    @staticmethod
    def _choose_overlapping_detectors(found_detectors: List[Tuple[int, Polygon]],
                                      point: Point) -> int:
        """ Find the detector which contains a given point located in the overlapping area between
        two adjacent detectors.

        :param found_detectors: a list of 2 tuples, each containing the detector index and
                                its footprint
        :param point: the point to locate inside the overlapping area between the 2 detectors
        :returns: the index of the detector containing the point.
        :raises ValueError: when not exactly 2 detectors are provided, or when the detectors do
                            not overlap, or when the point does not belong to their intersection.
        """
        if len(found_detectors) != 2:
            raise ValueError('Only 2 overlapping detectors supported. ')

        # compute the detectors intersection
        left_detector_polygon = found_detectors[0][1]
        right_detector_polygon = found_detectors[1][1]
        detectors_intersection = left_detector_polygon.intersection(right_detector_polygon)
        if detectors_intersection.is_empty:
            raise ValueError('Non overlapping S2 detectors')
        if not detectors_intersection.contains(point):
            raise ValueError(f'Point : {str(point)} not belonging to detectors intersection')

        # Build an horizontal line passing by the point and spanning the whole detectors width.
        x_min, _, x_max, _ = left_detector_polygon.union(right_detector_polygon).bounds
        line = LineString([(x_min, point.y), (x_max, point.y)])

        # Compute the intersection of this line with the detectors intersection and find its middle.
        x_line_min, _, x_line_max, _ = line.intersection(detectors_intersection).bounds
        x_middle = (x_line_min + x_line_max) / 2.

        # Select the right detector according to its position relative to the middle point.
        if point.x < x_middle:
            chosen_detector = found_detectors[0][0]
        else:
            chosen_detector = found_detectors[1][0]
        return chosen_detector

    @property
    def bands_detectors(self) -> Dict[str, List[Polygon]]:
        """ :returns: a dictionary whose keys are the S2 spectral bands ids and walues are fixed
                      size lists of polygons describing the footprints of the intersection of each
                      detector with a tile or None if this intersection is empty.
        """
        if self._bands_detectors is None:
            self._bands_detectors = {}
            detectors_footprint_files = self.get_detectors_footprints_files()
            for band_id, det_footprint_filepath in detectors_footprint_files.items():
                self._bands_detectors[band_id] = self.get_detectors_footprint(
                    det_footprint_filepath)
        return self._bands_detectors

    def get_detectors_footprints_files(self) -> Dict[str, Path]:
        """ :returns: a dictionary with S2 spectral band ids as keys and full path to the XML file
                      describing the detectors footprints for these spectral band.
        :raise ValueError: when the MTD_TL.xml file is ill-formed and cannot be read consistently.
        """
        # Parse the XML file
        tree = ET.parse(str(self.s2_metadata_path))
        root = tree.getroot()

        schema_location_key = '{http://www.w3.org/2001/XMLSchema-instance}schemaLocation'
        name_space = {'n1': root.attrib[schema_location_key].split()[0]}

        qualinfo = root.find('n1:Quality_Indicators_Info', name_space)
        if qualinfo is None:
            raise ValueError('Unable to find Quality_Indicators_Info element in XML metadata')

        pixlevel = qualinfo.find('Pixel_Level_QI')
        if pixlevel is None:
            raise ValueError('Unable to find Pixel_Level_QI element in XML metadata')

        qifiles = pixlevel.findall("./*[@type='MSK_DETFOO']")

        detectors_footprint_files = {}
        for qifile in qifiles:
            band_id = qifile.get('bandId')
            if band_id is None:
                raise ValueError('Error when retrieving band id for detector footprint file')
            relative_filepath = qifile.text
            if relative_filepath is None:
                raise ValueError('Error when retrieving detector footprint file name')
            detectors_footprint_files[S2_SPECTRAL_BANDS_NAMES[int(band_id)]] = \
                self.product_path / Path(relative_filepath)
        return detectors_footprint_files

    def get_detectors_footprint(self, det_footprint_filepath: Path) -> List[Optional[Polygon]]:
        """ Return the detectors footprints for a spectral band of an S2 product.

        :param det_footprint_filepath: path to the "MSK_DETFOO" XML file in the S2 product
                                       containing the detectors footprints of a spectral band.
        :returns: a fixed size list of detectors footprints expressed in the S2 product geometry if
                  a detector has a non empty intersection with the S2 tile footprint, or None
                  otherwise.
        :raise ValueError: when the XML file is ill-formed and cannot be read consistently.
        """
        detectors_footprints = [None] * NB_BANDS_DETECTORS

        if self.processing_baseline<400:
            # Before processing baseline up to 400, detfoo files are in format .gml
            # Parse the XML file
            tree = ET.parse(str(det_footprint_filepath))
            root = tree.getroot()
            name_space = {'gml': 'http://www.opengis.net/gml/3.2',
                        'eop': 'http://www.opengis.net/eop/2.0'}

            masks_members = root.findall('eop:maskMembers', name_space)
            if masks_members is None:
                raise ValueError(f'Cannot find any detectors '
                                 f'footprints in {det_footprint_filepath}')
            for masks_member in masks_members:
                mask_features = masks_member.findall('eop:MaskFeature', name_space)

                for feature in mask_features:
                    feat_id = feature.get(f'{{{name_space["gml"]}}}id')
                    if feat_id is None:
                        raise ValueError(f'Unable to find '
                                         f'mask_feature id in {det_footprint_filepath}')
                    detector_index = int(feat_id.split('-')[2]) - 1
                    vertices = feature.find(
                        './eop:extentOf/gml:Polygon/gml:exterior/gml:LinearRing/gml:posList',
                        name_space)
                    if vertices is None or vertices.text is None:
                        msg_err = f'Unable to find footprints coords in {det_footprint_filepath}'
                        raise ValueError(msg_err)
                    coords = [float(x) for x in vertices.text.split()]
                    nb_points = int(len(coords) / 3)
                    points = [(coords[3 * index], coords[3 * index + 1])
                              for index in range(nb_points)]
                    detectors_footprints[detector_index] = Polygon(points)
        else:
            # Since processing baseline 400, detfoo files are in format .jp2
            detfoo_raster = gdal.Open(str(det_footprint_filepath))
            band = detfoo_raster.GetRasterBand(1)

            srs = detfoo_raster.GetSpatialRef()
            driver = ogr.GetDriverByName("Memory")
            fs = driver.CreateDataSource("detfoo_memory.shp")

            layer = fs.CreateLayer("polygons", geom_type=ogr.wkbPolygon, srs=srs)
            fielddefn = ogr.FieldDefn('image_value', 0)
            layer.CreateField(fielddefn, 1)
            gdal.Polygonize(band, None, layer, 0, [], None)

            # Process the generated polygons
            for feature in layer:
                detector_index = feature.GetField("image_value")
                geom = feature.GetGeometryRef()

                if geom and geom.GetGeometryType() == ogr.wkbPolygon:
                    ring = geom.GetGeometryRef(0)
                    poly_coords = np.array(ring.GetPoints())

                    detectors_footprints[int(detector_index) - 1] = Polygon(poly_coords)

        return detectors_footprints

    def create_delta_time_provider(
            self, external_delta_times_path: Optional[Path] = None) -> S2DeltaTimeProvider:
        """ :returns: a delta time provider for this product
        """
        if external_delta_times_path is None or not external_delta_times_path.is_dir():
            excp_msg = 'Path to a directory containing the S2A and S2B delta times is required'
            raise ValueError(excp_msg)
        return S2DeltaTimeProvider(external_delta_times_path, self)
