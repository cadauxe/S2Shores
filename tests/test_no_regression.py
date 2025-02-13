import subprocess
from tests.conftest import S2SHORESTestsPath

def test_nominal_spatialCorrelation_s2(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    ###TODO
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
    ###TODO
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
#     ###TODO
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
    ###TODO
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
    ###TODO
    """
    cmd = (f"python3 {s2shores_paths.cli_path} --input_product {s2shores_paths.swach7_product_dir} --output_dir {s2shores_paths.output_dir} "
           f"--config_file {s2shores_paths.config_dir}/config7/{s2shores_paths.yaml_file} --product_type geotiff "
           f"--debug_path {s2shores_paths.debug} --debug_file {s2shores_paths.debug}/debug_points_SWASH_7_4.yaml")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    out, _ = proc.communicate()
    assert proc.returncode == 0, out.decode('utf-8')
    ####TEST the ouputs.....


def test_debug_pointswach_spatial_dft(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    ###TODO
    """
    cmd = (f"python3 {s2shores_paths.cli_path} --input_product {s2shores_paths.swach8_product_dir} --output_dir {s2shores_paths.output_dir} "
           f"--config_file {s2shores_paths.config_dir}/config5/{s2shores_paths.yaml_file} --product_type geotiff "
           f"--debug_path {s2shores_paths.debug} --debug_file {s2shores_paths.debug}/debug_points_SWASH_8_2.yaml")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    out, _ = proc.communicate()
    assert proc.returncode == 0, out.decode('utf-8')
    ####TEST the ouputs.....


def test_debug_pointswach_spatial_corr(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    ###TODO
    """
    cmd = (f"python3 {s2shores_paths.cli_path} --input_product {s2shores_paths.swach8_product_dir} --output_dir {s2shores_paths.output_dir} "
           f"--config_file {s2shores_paths.config_dir}/config6/{s2shores_paths.yaml_file} --product_type geotiff "
           f"--debug_path {s2shores_paths.debug} --debug_file {s2shores_paths.debug}/debug_points_SWASH_8_2.yaml")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    out, _ = proc.communicate()
    assert proc.returncode == 0, out.decode('utf-8')
    ####TEST the ouputs.....


def test_limitroi_s2(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    ###TODO
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
    ###TODO
    """
    cmd = (f"python3 {s2shores_paths.cli_path} --input_product {s2shores_paths.s2new_product_dir} --output_dir {s2shores_paths.output_dir} "
           f"--config_file {s2shores_paths.config_dir}/config8/{s2shores_paths.yaml_file} --delta_times_dir {s2shores_paths.delta_times_dir} --product_type S2 "
           f"--distoshore_file {s2shores_paths.dis2shore_dir}/disToShore_30TXR.TIF "
           f"--nb_subtiles 36 --debug_path {s2shores_paths.debug} --debug_file {s2shores_paths.debug}/debug_points_30TXR_notongrid.yaml")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    out, _ = proc.communicate()
    assert proc.returncode == 0, out.decode('utf-8')
    ####TEST the ouputs.....

def test_debug_area_funwave(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    ###TODO
    """
    cmd = (f"python3 {s2shores_paths.cli_path} --input_product {s2shores_paths.funwave_product_dir} --output_dir {s2shores_paths.output_dir} "
           f"--config_file {s2shores_paths.config_dir}/config9/{s2shores_paths.yaml_file} --product_type geotiff "
           f"--debug_path {s2shores_paths.debug} --debug_file {s2shores_paths.debug}/debug_area_funwave.yaml")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    out, _ = proc.communicate()
    assert proc.returncode == 0, out.decode('utf-8')
    ####TEST the ouputs.....

def test_roi_profiling_s2(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    ###TODO
    """
    cmd = (f"python3 {s2shores_paths.cli_path} --input_product {s2shores_paths.s2old_product_dir} --output_dir {s2shores_paths.output_dir} "
           f"--config_file {s2shores_paths.config_dir}/config2/{s2shores_paths.yaml_file} --product_type S2 "
           f"--delta_times_dir {s2shores_paths.delta_times_dir}"
           f"--distoshore_file {s2shores_paths.dis2shore_dir}/GMT_intermediate_coast_distance_01d_test_5000.nc"
           f"--nb_subtiles 36 --roi_file {s2shores_paths.roi_dir}/30TXR-ROI.shp --profiling")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    out, _ = proc.communicate()
    assert proc.returncode == 0, out.decode('utf-8')
    ####TEST the ouputs.....