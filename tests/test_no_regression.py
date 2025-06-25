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
import shutil

from tests.test_utils import compare_files
from click.testing import CliRunner

from s2shores.bathylauncher.bathy_processing import process_command


S2NEW_FILE = "S2A_MSIL1C_20200622T105631_N0500_R094_T30TXR_20231110T094313.SAFE"
S2OLD_FILE = "S2A_MSIL1C_20200622T105631_N0209_R094_T30TXR_20200622T130553.SAFE"
FUNWAVE_FILE = "funwave.tif"
PARAMS_FILE = "wave_bathy_inversion_config.yaml"
DISTOSHORE_FILE_NC = "GMT_intermediate_coast_distance_01d_test_5000.nc"
DISTOSHORE_FILE_TIF = "disToShore_30TXR.TIF"
SWASH_8_2_FILE = "testcase_8_2.tif"
SWASH_7_4_FILE = "testcase_7_4.tif"
PNEO_FILE = "Duck_PNEO_XS_b3_VT.tif"


def test_nominal_tri_stereo_pneo(test_path) -> None:
    """
    Test PNEO data without ROI and distoshore, with geotiff
    product, nb_subtiles=1 and Layers-type debug.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.

    """

    output_path = f'{test_path}/output/nominal_tri_stereo_pneo'
    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', f'{test_path}/products/PNEO_DUCK/{PNEO_FILE}',
        '--product_type', 'geotiff',
        '--output_dir',  f'{output_path}',
        '--config_file', f'{test_path}/config/config1/{PARAMS_FILE}',
	'--roi_file', f'{test_path}/ROI/PNEO-DuckROI.shp',
	'--limit_to_roi'])
    print(result.output)
    compare_files(reference_dir=f'{test_path}/reference_results/nominal_tri_stereo_pneo',
                  output_dir=f'{output_path}')


def test_nominal_video(test_path) -> None:
    """
    Test Funwave data without ROI and distoshore, with
    geotiff product, nb_subtiles=1 and Layers-type debug.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """

    output_path = f'{test_path}/output/nominal_video'
    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', f'{test_path}/products/FUNWAVE/{FUNWAVE_FILE}',
        '--product_type', 'geotiff',
        '--output_dir', f'{output_path}',
        '--config_file', f'{test_path}/config/config4/{PARAMS_FILE}',
        '--nb_subtiles', '4'])
    print(result.output)
    compare_files(reference_dir=f'{test_path}/reference_results/nominal_video',
                  output_dir=f'{output_path}')


def test_debug_pointswash_temporal_corr(test_path) -> None:
    """
    Test SWASH7.4 data without ROI, with geotiff product, temporal
    correlation debug, grid debug point mode and Layers-type expert.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    output_path = f'{test_path}/output/debug_pointswash_temporal_corr'
    debug_path = f'{output_path}/debug'
    if os.path.isdir(debug_path) :
        shutil.rmtree(debug_path)
    os.mkdir(debug_path)

    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', f'{test_path}/products/SWASH_7_4/{SWASH_7_4_FILE}',
        '--product_type', 'geotiff',
        '--output_dir', f'{output_path}',
        '--config_file', f'{test_path}/config/config7/{PARAMS_FILE}',
        '--debug_path', f'{debug_path}',
        '--debug_file', f'{test_path}/debug/debug_points_SWASH_7_4.yaml'])
    print(result.output)
    compare_files(reference_dir=f'{test_path}/reference_results/debug_pointswash_temporal_corr',
                  output_dir=f'{output_path}',
                  debug_dir = f'{debug_path}')


def test_debug_pointswash_spatial_dft(test_path) -> None:
    """
    Test SWASH8.2 data without ROI, with geotiff product
    , dft spatial debug and grid debug point mode.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    output_path = f'{test_path}/output/debug_pointswash_spatial_dft'
    debug_path = f'{output_path}/debug'
    if os.path.isdir(debug_path):
        shutil.rmtree(debug_path)
    os.mkdir(debug_path)

    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', f'{test_path}/products/SWASH_8_2/{SWASH_8_2_FILE}',
        '--product_type', 'geotiff',
        '--output_dir', f'{output_path}',
        '--config_file', f'{test_path}/config/config5/{PARAMS_FILE}',
        '--debug_path', f'{debug_path}',
        '--debug_file', f'{test_path}/debug/debug_points_SWASH_8_2.yaml'])
    print(result.output)
    compare_files(reference_dir=f'{test_path}/reference_results/debug_pointswash_spatial_dft',
                  output_dir= f'{output_path}',
                  debug_dir = f'{debug_path}')


def test_debug_pointswash_spatial_corr(test_path) -> None:
    """
    Test SWASH8.2 data without ROI, with geotiff product, spatial
    correlation debug, grid debug point mode and Layers-type nominal.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    output_path = f'{test_path}/output/debug_pointswash_spatial_corr'
    debug_path = f'{output_path}/debug'
    if os.path.isdir(debug_path) :
        shutil.rmtree(debug_path)
    os.mkdir(debug_path)
    
    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', f'{test_path}/products/SWASH_8_2/{SWASH_8_2_FILE}',
        '--product_type', 'geotiff',
        '--output_dir', f'{output_path}',
        '--config_file', f'{test_path}/config/config6/{PARAMS_FILE}',
        '--debug_path', f'{debug_path}',
        '--debug_file', f'{test_path}/debug/debug_points_SWASH_8_2.yaml'])
    print(result.output)
    compare_files(reference_dir=f'{test_path}/reference_results/debug_pointswash_spatial_corr',
                  output_dir = f'{output_path}',
                  debug_dir = f'{debug_path}')


def test_limitroi_s2(test_path) -> None:
    """
    Test Sentinel-2 30TXR New data with ROI, ROI limit and sequential option.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', f'{test_path}/products/S2_30TXR_NEW/{S2NEW_FILE}',
        '--output_dir', f'{test_path}/output/limitroi_s2',
        '--config_file', f'{test_path}/config/config2/{PARAMS_FILE}',
        '--delta_times_dir', f'{test_path}/deltatimesS2/esa',
        '--distoshore_file', f'{test_path}/distoshore/{DISTOSHORE_FILE_TIF}',
        '--product_type', 'S2',
        '--nb_subtiles', '36',
        '--roi_file', f'{test_path}/ROI/30TXR-ROI.shp',
        '--limit_to_roi',
        '--sequential'])
    print(result.output)
    compare_files(reference_dir=f'{test_path}/reference_results/limitroi_s2',
                  output_dir= f'{test_path}/output/limitroi_s2')


def test_debug_mode_point_s2(test_path) -> None:
    """
    Test Sentinel-2 30TXR New data with S2 product and point debug point mode.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    output_path = f'{test_path}/output/debug_mode_point_s2'
    debug_path = f'{output_path}/debug'
    if os.path.isdir(debug_path) :
        shutil.rmtree(debug_path)
    os.mkdir(debug_path)

    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', f'{test_path}/products/S2_30TXR_NEW/{S2NEW_FILE}',
        '--product_type', 'S2',
        '--output_dir', f'{output_path}',
        '--config_file', f'{test_path}/config/config8/{PARAMS_FILE}',
        '--delta_times_dir', f'{test_path}/deltatimesS2/cnes',
        '--distoshore_file', f'{test_path}/distoshore/{DISTOSHORE_FILE_TIF}',
        '--nb_subtiles', '36',
        '--debug_path', f'{debug_path}',
        '--debug_file', f'{test_path}/debug/debug_points_30TXR_notongrid.yaml'])
    print(result.output)
    compare_files(reference_dir = f'{test_path}/reference_results/debug_mode_point_s2',
                  output_dir = f'{output_path}',
                  debug_dir = f'{debug_path}')


def test_debug_area_funwave(test_path) -> None:
    """
    Test Funwave data with geotiff product and debug area.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """

    output_path = f'{test_path}/output/debug_area_funwave'
    debug_path = f'{output_path}/debug'
    if os.path.isdir(debug_path) :
        shutil.rmtree(debug_path)
    os.mkdir(debug_path)
    
    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', f'{test_path}/products/FUNWAVE/{FUNWAVE_FILE}',
        '--product_type', 'geotiff',
        '--output_dir', f'{output_path}',
        '--config_file', f'{test_path}/config/config9/{PARAMS_FILE}',
        '--debug_path', f'{debug_path}',
        '--debug_file', f'{test_path}/debug/debug_area_funwave.yaml'])
    print(result.output)
    compare_files(reference_dir=f'{test_path}/reference_results/debug_area_funwave',
                  output_dir=f'{output_path}',
                  debug_dir = f'{debug_path}')

def test_roi_profiling_s2(test_path, capsys) -> None:
    """
    Test Sentinel-2 30TXR Old data without ROI limit
    , with S2 product, ROI and profiling option.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    output_path = f'{test_path}/output/roi_profiling_s2'
    profiling_path = f'{output_path}/profiling'
    if os.path.isdir(profiling_path) :
        shutil.rmtree(profiling_path)
    os.mkdir(profiling_path)

    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', f'{test_path}/products/S2_30TXR_OLD/{S2OLD_FILE}',
        '--product_type', 'S2',
        '--output_dir', f'{output_path}',
        '--config_file', f'{test_path}/config/config2/{PARAMS_FILE}',
        '--distoshore_file', f'{test_path}/distoshore/{DISTOSHORE_FILE_NC}',
        '--delta_times_dir', f'{test_path}/deltatimesS2/esa',
        '--roi_file', f'{test_path}/ROI/30TXR-ROI.shp',
        '--nb_subtiles', '36', 
        '--profiling'])
    print(result.output)
    captured = capsys.readouterr()
    with open(f'{profiling_path}/profiling.txt', 'w') as p:
        p.write(captured.out)

    # no comparison of profiling files
    compare_files(reference_dir=f"{test_path}/reference_results/roi_profiling_s2",
                  output_dir=f'{output_path}')


def test_nominal_dft_s2_cnes_deltat(test_path) -> None:
    """
    Test Sentinel-2 30TXR New data without ROI, with S2 product,
    nb_subtiles>1, Layers-type debug and tile distoshore.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', f'{test_path}/products/S2_30TXR_NEW/{S2NEW_FILE}',
        '--product_type', 'S2',
        '--output_dir', f'{test_path}/output/nominal_dft_s2_cnes_deltat',
        '--config_file', f'{test_path}/config/config3/{PARAMS_FILE}',
        '--delta_times_dir', f'{test_path}/deltatimesS2/cnes',
        '--distoshore_file',  f'{test_path}/distoshore/{DISTOSHORE_FILE_TIF}',
        '--nb_subtiles', '36'])
    print(result.output)
    compare_files(reference_dir=f"{test_path}/reference_results/nominal_dft_s2_cnes_deltat",
                  output_dir=f'{test_path}/output/nominal_dft_s2_cnes_deltat')

def test_nominal_spatialcorr_s2_cnes_deltat(test_path) -> None:
    """
    Test Sentinel-2 30TXR New data without ROI, with S2 product,
    nb_subtiles>1, Layers-type debug and global distoshore.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """

    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', f'{test_path}/products/S2_30TXR_NEW/{S2NEW_FILE}',
        '--product_type', 'S2',
        '--output_dir', f'{test_path}/output/nominal_spatialcorr_s2_cnes_deltat',
        '--config_file', f'{test_path}/config/config2/{PARAMS_FILE}',
        '--delta_times_dir', f'{test_path}/deltatimesS2/cnes',
        '--distoshore_file', f'{test_path}/distoshore/{DISTOSHORE_FILE_NC}',
        '--nb_subtiles', '36'])
    print(result.output)
    compare_files(reference_dir=f"{test_path}/reference_results/nominal_spatialcorr_s2_cnes_deltat",
                  output_dir=f'{test_path}/output/nominal_spatialcorr_s2_cnes_deltat')
