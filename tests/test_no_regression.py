import subprocess
import os
import glob
import xarray as xr
from tests.conftest import S2SHORESTestsPath

def compare_files(reference_dir : str):
    """
    ###TODO
    """
    # Get all directories in the specified parent directory
    dirs = [d for d in glob.glob(os.path.join(parent_dir, "*/")) if os.path.isdir(d)]

    # Find the most recently created directory, ie. the test output directory
    out_test_dir = max(dirs, key=os.path.getctime)

    ref_files = os.listdir(reference_dir)
    out_test_files = os.listdir(out_test_dir)

    if ref_files == out_test_files:
        print("Both directories contain the same filenames.")
        return True
    else:
        raise Exception("Filenames differ between the directories.\n"
               f"Only in {reference_dir} : {reference_dir} - {out_test_dir}"
               f"Only in {out_test_dir} : {out_test_dir} - {reference_dir}")



    #Assert the files in the reference directory are the same than the ones in the lastly created directory
    ref_nc = [nc_file for nc_file in ref_files if ".nc" in nc_file]
    out_nc = [nc_file for nc_file in out_test_files if ".nc" in nc_file]

    ref_xr = xr.open_dataset(reference_dir / ref_nc)
    out_xr = xr.open_dataset(out_test_dir / out_nc)

    xr.testing.assert_equal(ref_xr, out_xr)


def test_nominal_spatialCorrelation_s2(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    PASSED
    Test Sentinel-2 30TXR Old data without ROI, with S2 product, nb_subtiles>1, Layers-type debug and global distoshore.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    cmd = (f"python3 {s2shores_paths.cli_path} --input_product {s2shores_paths.s2old_product_dir} --output_dir {s2shores_paths.output_dir} "
           f"--config_file {s2shores_paths.config_dir}/config2/{s2shores_paths.yaml_file} --delta_times_dir {s2shores_paths.delta_times_dir} --product_type S2 "
           f"--distoshore_file {s2shores_paths.dis2shore_dir}/GMT_intermediate_coast_distance_01d_test_5000.nc --nb_subtiles 36")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    out, _ = proc.communicate()
    assert proc.returncode == 0, out.decode('utf-8')
    ####TEST the ouputs.....


def test_nominal_dft_s2(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    PASSED
    Test Sentinel-2 30TXR New data without ROI, with S2 product, nb_subtiles>1, Layers-type debug and tile distoshore.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    cmd = (f"python3 {s2shores_paths.cli_path} --input_product {s2shores_paths.s2new_product_dir} --output_dir {s2shores_paths.output_dir} "
           f"--config_file {s2shores_paths.config_dir}/config3/{s2shores_paths.yaml_file} --delta_times_dir {s2shores_paths.delta_times_dir} --product_type S2 "
           f"--distoshore_file {s2shores_paths.dis2shore_dir}/disToShore_30TXR.TIF --nb_subtiles 36")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    out, _ = proc.communicate()
    assert proc.returncode == 0, out.decode('utf-8')
    ####TEST the ouputs.....


# def test_nominal_tri_stereo_pneo(s2shores_paths: S2SHORESTestsPath) -> None:
#     """
#     Test PNEO data without ROI and distoshore, with geotiff product, nb_subtiles=1 and Layers-type debug.
#
#     - Verify that all expected output files are created.
#     - Ensure the generated .nc output file matches the reference.
#     """
#     cmd = (f"python3 {s2shores_paths.cli_path} --input_product {s2shores_paths.pneo_product_dir} --output_dir {s2shores_paths.output_dir} "
#            f"--config_file {s2shores_paths.config_dir}/config1/{s2shores_paths.yaml_file} --product_type geotiff "
#            f"--nb_subtiles 36")
#     proc = subprocess.Popen(
#         cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
#     )
#     out, _ = proc.communicate()
#     assert proc.returncode == 0, out.decode('utf-8')
#     ####TEST the ouputs.....


def test_nominal_video(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    PASSED
    Test Funwave data without ROI and distoshore, with geotiff product, nb_subtiles=1 and Layers-type debug.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    cmd = (f"python3 {s2shores_paths.cli_path} --input_product {s2shores_paths.funwave_product_dir} --output_dir {s2shores_paths.output_dir} "
           f"--config_file {s2shores_paths.config_dir}/config4/{s2shores_paths.yaml_file} --product_type geotiff "
           f"--nb_subtiles 4")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    out, _ = proc.communicate()
    assert proc.returncode == 0, out.decode('utf-8')
    ####TEST the ouputs.....


def test_debug_pointswach_temporal_corr(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    PASSED
    Test SWASH7.4 data without ROI, with geotiff product, temporal correlation debug, grid debug point mode and Layers-type expert.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    cmd = (f"python3 {s2shores_paths.cli_path} --input_product {s2shores_paths.swach7_product_dir} --output_dir {s2shores_paths.output_dir} "
           f"--config_file {s2shores_paths.config_dir}/config7/{s2shores_paths.yaml_file} --product_type geotiff "
           f"--debug_path {s2shores_paths.debug_dir} --debug_file {s2shores_paths.debug_dir}/debug_points_SWASH_7_4.yaml")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    out, _ = proc.communicate()
    assert proc.returncode == 0, out.decode('utf-8')
    ####TEST the ouputs.....


def test_debug_pointswach_spatial_dft(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    FAILED
    Test SWASH8.2 data without ROI, with geotiff product, dft spatial debug and grid debug point mode.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    cmd = (f"python3 {s2shores_paths.cli_path} --input_product {s2shores_paths.swach8_product_dir} --output_dir {s2shores_paths.output_dir} "
           f"--config_file {s2shores_paths.config_dir}/config5/{s2shores_paths.yaml_file} --product_type geotiff "
           f"--debug_path {s2shores_paths.debug_dir} --debug_file {s2shores_paths.debug_dir}/debug_points_SWASH_8_2.yaml")
    print(cmd)
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    out, _ = proc.communicate()
    assert proc.returncode == 0, out.decode('utf-8')
    ####TEST the ouputs.....


def test_debug_pointswach_spatial_corr(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    FAILED
    Test SWASH8.2 data without ROI, with geotiff product, spatial correlation debug, grid debug point mode and Layers-type nominal.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    cmd = (f"python3 {s2shores_paths.cli_path} --input_product {s2shores_paths.swach8_product_dir} --output_dir {s2shores_paths.output_dir} "
           f"--config_file {s2shores_paths.config_dir}/config6/{s2shores_paths.yaml_file} --product_type geotiff "
           f"--debug_path {s2shores_paths.debug_dir} --debug_file {s2shores_paths.debug_dir}/debug_points_SWASH_8_2.yaml")
    print(cmd)
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    out, _ = proc.communicate()
    assert proc.returncode == 0, out.decode('utf-8')
    ####TEST the ouputs.....


def test_limitroi_s2(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    PASSED
    Test Sentinel-2 30TXR New data with ROI, ROI limit and sequential option.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    cmd = (f"python3 {s2shores_paths.cli_path} --input_product {s2shores_paths.s2new_product_dir} --output_dir {s2shores_paths.output_dir} "
           f"--config_file {s2shores_paths.config_dir}/config2/{s2shores_paths.yaml_file} --delta_times_dir {s2shores_paths.delta_times_dir} --product_type S2 "
           f"--distoshore_file {s2shores_paths.dis2shore_dir}/disToShore_30TXR.TIF "
           f"--nb_subtiles 36 --roi_file {s2shores_paths.roi_dir}/30TXR-ROI.shp --limit_to_roi --sequential")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    out, _ = proc.communicate()
    assert proc.returncode == 0, out.decode('utf-8')
    ####TEST the ouputs.....

def test_debug_mode_point_s2(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    FAILED
    Test Sentinel-2 30TXR New data with S2 product and point debug point mode.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    cmd = (f"python3 {s2shores_paths.cli_path} --input_product {s2shores_paths.s2new_product_dir} --output_dir {s2shores_paths.output_dir} "
           f"--config_file {s2shores_paths.config_dir}/config8/{s2shores_paths.yaml_file} --delta_times_dir {s2shores_paths.delta_times_dir} --product_type S2 "
           f"--distoshore_file {s2shores_paths.dis2shore_dir}/disToShore_30TXR.TIF "
           f"--nb_subtiles 36 --debug_path {s2shores_paths.debug_dir} --debug_file {s2shores_paths.debug_dir}/debug_points_30TXR_notongrid.yaml")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    out, _ = proc.communicate()
    assert proc.returncode == 0, out.decode('utf-8')
    ####TEST the ouputs.....

def test_debug_area_funwave(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    FAILED
    Test Funwave data with geotiff product and debug area.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    cmd = (f"python3 {s2shores_paths.cli_path} --input_product {s2shores_paths.funwave_product_dir} --output_dir {s2shores_paths.output_dir} "
           f"--config_file {s2shores_paths.config_dir}/config9/{s2shores_paths.yaml_file} --product_type geotiff "
           f"--debug_path {s2shores_paths.debug_dir} --debug_file {s2shores_paths.debug_dir}/debug_area_funwave.yaml")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    out, _ = proc.communicate()
    assert proc.returncode == 0, out.decode('utf-8')
    ####TEST the ouputs.....

def test_roi_profiling_s2(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    PASSED
    Test Sentinel-2 30TXR Old data without ROI limit, with S2 product, ROI and profiling option.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    cmd = (f"python3 {s2shores_paths.cli_path} --input_product {s2shores_paths.s2old_product_dir} --output_dir {s2shores_paths.output_dir} "
           f"--config_file {s2shores_paths.config_dir}/config2/{s2shores_paths.yaml_file} --product_type S2 "
           f"--delta_times_dir {s2shores_paths.delta_times_dir} "
           f"--distoshore_file {s2shores_paths.dis2shore_dir}/GMT_intermediate_coast_distance_01d_test_5000.nc "
           f"--nb_subtiles 36 --roi_file {s2shores_paths.roi_dir}/30TXR-ROI.shp --profiling")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    out, _ = proc.communicate()
    assert proc.returncode == 0, out.decode('utf-8')
    ####TEST the ouputs.....