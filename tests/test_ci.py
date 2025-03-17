# -*- coding: utf-8 -*-
"""
Tests to ensure no code regression, the outputs are compared to reference results.


:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2024 CNES. All rights reserved.
:license: see LICENSE file
:created: 06/03/2025

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
import os
import glob
import zipfile
import xarray as xr
from click.testing import CliRunner
from tests.conftest import S2SHORESTestsPath

from s2shores.bathylauncher.bathy_processing import process_command

def compare_files(reference_dir : str, output_dir : str):
    """
    Compares the contents of the reference directory with the most recently created
    test output directory. Ensures that the filenames match and that the contents of
    NetCDF files are identical.

    :param reference_dir: The directory containing reference files.
    :returns: True if the directories have identical filenames and matching NetCDF content.
    :raises Exception: If filenames differ between the directories or NetCDF file contents do not match.
    """
    # Get all directories in the specified parent directory
    dirs = [d for d in glob.glob(os.path.join(output_dir, "*/")) if os.path.isdir(d)]

    # Find the most recently created directory, ie. the test output directory
    out_test_dir = max(dirs, key=os.path.getctime)

    ref_files = os.listdir(reference_dir)
    out_test_files = os.listdir(out_test_dir)

    if ref_files == out_test_files:
        print("Both directories contain the same filenames.")
    else:
        raise Exception("Filenames differ between the directories.\n"
               f"Only in {reference_dir} : {[item for item in ref_files if item not in out_test_files]}\n"
               f"Only in {out_test_dir} : {[item for item in out_test_files if item not in ref_files]}")

    #Assert the files in the reference directory are the same
    #than the ones in the lastly created directory
    for nc_file in ref_files :
        if ".nc" in nc_file :
            ref_nc = nc_file
    for nc_file in out_test_files:
        if ".nc" in nc_file:
            out_nc = nc_file

    ref_xr = xr.open_dataset(os.path.join(reference_dir, ref_nc))
    out_xr = xr.open_dataset(os.path.join(out_test_dir, out_nc))

    xr.testing.assert_equal(ref_xr, out_xr)


def unzip_file(zip_path):
    """
    Unzips a file and extracts its contents into the same directory.

    :param zip_path: Path to the ZIP file.
    :returns: List of extracted file paths.
    :raises FileNotFoundError: If the ZIP file does not exist.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file '{zip_path}' not found.")

    extract_to = os.path.dirname(zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        extracted_files = [os.path.join(extract_to, f) for f in zip_ref.namelist()]
    return extracted_files


def test_nominal_spatialCorrelation_s2_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Sentinel-2 30TXR Old data without ROI, with S2 product,
    nb_subtiles>1, Layers-type debug and global distoshore.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    dis2shore_file = "GMT_intermediate_coast_distance_01d_test_5000_cropped.nc"
    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.s2new_cropped),
        '--product_type', 'S2',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config2/wave_bathy_inversion_config_quick.yaml',
        '--distoshore_file', f'{s2shores_paths.dis2shore_dir}/{dis2shore_file}',
        '--delta_times_dir', str(s2shores_paths.delta_times_dir),
        '--nb_subtiles', '36'])

    print(result.output)
    compare_files(reference_dir = f"{s2shores_paths.output_dir}/CI_tests/nominal_spatialCorrelation_s2_quick",
                  output_dir = s2shores_paths.output_dir)


def test_nominal_dft_s2_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Sentinel-2 30TXR New data without ROI, with S2 product,
    nb_subtiles>1, Layers-type debug and tile distoshore.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.s2new_cropped),
        '--product_type', 'S2',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config3/wave_bathy_inversion_config_quick.yaml',
        '--distoshore_file', f'{s2shores_paths.dis2shore_dir}/disToShore_30TXR_cropped.TIF',
        '--delta_times_dir', str(s2shores_paths.delta_times_dir),
        '--nb_subtiles', '36'])

    print(result.output)
    compare_files(reference_dir = f"{s2shores_paths.output_dir}/CI_tests/nominal_dft_s2_quick",
                  output_dir = s2shores_paths.output_dir)


def test_nominal_video_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Funwave data without ROI and distoshore, with
    geotiff product, nb_subtiles=1 and Layers-type debug.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    unzip_file(s2shores_paths.funwave_cropped.with_suffix('.zip'))
    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.funwave_cropped),
        '--product_type', 'geotiff',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config4/wave_bathy_inversion_config_quick.yaml',
        '--nb_subtiles', '4'])

    print(result.output)
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/nominal_video_quick",
                  output_dir=s2shores_paths.output_dir)


def test_debug_pointswash_temporal_corr_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test SWASH7.4 data without ROI, with geotiff product, temporal
    correlation debug, grid debug point mode and Layers-type expert.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    if not(os.path.isdir(f'{s2shores_paths.output_dir}/debug_pointswash_temporal_corr')) :
        os.mkdir(f'{s2shores_paths.output_dir}/debug_pointswash_temporal_corr')

    unzip_file(s2shores_paths.swash7_cropped.with_suffix('.zip'))
    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.swash7_cropped),
        '--product_type', 'geotiff',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config7/wave_bathy_inversion_config_quick.yaml',
        '--debug_path', f'{s2shores_paths.output_dir}/debug_pointswash_temporal_corr',
        '--debug_file', f'{s2shores_paths.debug_dir}/debug_points_SWASH_7_4_cropped.yaml'])

    print(result.output)
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/debug_pointswash_temporal_corr_quick",
                  output_dir=s2shores_paths.output_dir)


def test_debug_pointswash_spatial_dft_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test SWASH8.2 data without ROI, with geotiff product, spatial dft debug and grid debug point mode.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    if not(os.path.isdir(f'{s2shores_paths.output_dir}/debug_pointswash_spatial_dft')) :
        os.mkdir(f'{s2shores_paths.output_dir}/debug_pointswash_spatial_dft')

    unzip_file(s2shores_paths.swash8_cropped.with_suffix('.zip'))
    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.swash8_cropped),
        '--product_type', 'geotiff',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config5/wave_bathy_inversion_config_quick.yaml',
        '--debug_path', f'{s2shores_paths.output_dir}/debug_pointswash_spatial_dft',
        '--debug_file', f'{s2shores_paths.debug_dir}/debug_points_SWASH_8_2_cropped.yaml'])

    print(result.output)
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/debug_pointswash_dft_quick",
                  output_dir=s2shores_paths.output_dir)


def test_debug_pointswash_spatial_corr_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test SWASH8.2 data without ROI, with geotiff product, spatial
    correlation debug, grid debug point mode and Layers-type nominal.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    if not(os.path.isdir(f'{s2shores_paths.output_dir}/debug_pointswash_spatial_corr')) :
        os.mkdir(f'{s2shores_paths.output_dir}/debug_pointswash_spatial_corr')

    unzip_file(s2shores_paths.swash8_cropped.with_suffix('.zip'))
    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.swash8_cropped),
        '--product_type', 'geotiff',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config6/wave_bathy_inversion_config_quick.yaml',
        '--debug_path', f'{s2shores_paths.output_dir}/debug_pointswash_spatial_corr',
        '--debug_file', f'{s2shores_paths.debug_dir}/debug_points_SWASH_8_2_cropped.yaml'])

    print(result.output)
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/debug_pointswash_spatial_corr_quick",
                  output_dir=s2shores_paths.output_dir)


def test_limitroi_s2_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Sentinel-2 30TXR New data with ROI, ROI limit and sequential option.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.s2new_cropped),
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config2/wave_bathy_inversion_config_quick.yaml',
        '--delta_times_dir', str(s2shores_paths.delta_times_dir),
        '--distoshore_file', f'{s2shores_paths.dis2shore_dir}/disToShore_30TXR_cropped.TIF',
        '--product_type', 'S2',
        '--nb_subtiles', '36',
        '--roi_file', f'{s2shores_paths.roi_dir}/30TXR-ROI-cropped.shp',
        '--limit_to_roi',
        '--sequential'])

    print(result.output)
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/limitroi_s2_quick",
                  output_dir=s2shores_paths.output_dir)


def test_debug_mode_point_s2_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Sentinel-2 30TXR New data with S2 product and point debug point mode.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    if not(os.path.isdir(f'{s2shores_paths.output_dir}/debug_mode_point_s2')) :
        os.mkdir(f'{s2shores_paths.output_dir}/debug_mode_point_s2')

    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.s2new_cropped),
        '--product_type', 'S2',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config8/wave_bathy_inversion_config_quick.yaml',
        '--delta_times_dir', str(s2shores_paths.delta_times_dir),
        '--distoshore_file', f'{s2shores_paths.dis2shore_dir}/disToShore_30TXR_cropped.TIF',
        '--nb_subtiles', '36',
        '--debug_path', f'{s2shores_paths.output_dir}/debug_mode_point_s2',
        '--debug_file', f'{s2shores_paths.debug_dir}/debug_points_30TXR_notongrid_cropped.yaml'])

    print(result.output)
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/debug_mode_point_s2_quick",
                  output_dir=s2shores_paths.output_dir)



def test_debug_area_funwave_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Funwave data with geotiff product and debug area.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    if not(os.path.isdir(f'{s2shores_paths.output_dir}/debug_area_funwave')) :
        os.mkdir(f'{s2shores_paths.output_dir}/debug_area_funwave')

    unzip_file(s2shores_paths.funwave_cropped.with_suffix('.zip'))
    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.funwave_cropped),
        '--product_type', 'geotiff',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config9/wave_bathy_inversion_config_quick.yaml',
        '--debug_path', f'{s2shores_paths.output_dir}/debug_area_funwave',
        '--debug_file', f'{s2shores_paths.debug_dir}/debug_area_funwave_cropped.yaml'])

    print(result.output)
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/debug_area_funwave_quick",
                   output_dir=s2shores_paths.output_dir)


def test_roi_profiling_s2_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Sentinel-2 30TXR Old data without ROI limit
    , with S2 product, ROI and profiling option.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    dis2shore_file = ("GMT_intermediate_coast_distance_01d_test_5000_cropped.nc")

    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.s2new_cropped),
        '--product_type', 'S2',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config2/wave_bathy_inversion_config_quick.yaml',
        '--distoshore_file', f'{s2shores_paths.dis2shore_dir}/{dis2shore_file}',
        '--delta_times_dir', str(s2shores_paths.delta_times_dir),
        '--roi_file', f'{s2shores_paths.roi_dir}/30TXR-ROI-cropped.shp',
        '--nb_subtiles', '36',
        '--profiling'])

    print(result.output)
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/roi_profiling_s2_quick",
                  output_dir=s2shores_paths.output_dir)
