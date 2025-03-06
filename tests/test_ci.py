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
        return True
    else:
        raise Exception("Filenames differ between the directories.\n"
               f"Only in {reference_dir} : {reference_dir} - {out_test_dir}"
               f"Only in {out_test_dir} : {out_test_dir} - {reference_dir}")


    #Assert the files in the reference directory are the same
    #than the ones in the lastly created directory
    ref_nc = [nc_file for nc_file in ref_files if ".nc" in nc_file]
    out_nc = [nc_file for nc_file in out_test_files if ".nc" in nc_file]

    ref_xr = xr.open_dataset(reference_dir / ref_nc)
    out_xr = xr.open_dataset(out_test_dir / out_nc)

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


def test_nominal_video_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Funwave data without ROI and distoshore, with
    geotiff product, nb_subtiles=1 and Layers-type debug.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    unzip_file(s2shores_paths.funwave_cropped.with_suffix('.zip'))
    runner = CliRunner()

    runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.funwave_cropped),
        '--product_type', 'geotiff',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config4/wave_bathy_inversion_config_quick.yaml',
        '--nb_subtiles', '4'])
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/nominal_video_quick_8s",
                  output_dir=s2shores_paths.output_dir)


def test_debug_pointswach_temporal_corr_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test SWASH7.4 data without ROI, with geotiff product, temporal
    correlation debug, grid debug point mode and Layers-type expert.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    unzip_file(s2shores_paths.swach7_cropped.with_suffix('.zip'))
    runner = CliRunner()

    runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.swach7_cropped),
        '--product_type', 'geotiff',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config7/wave_bathy_inversion_config_quick.yaml',
        '--debug_path', f'{s2shores_paths.output_dir}/debug_pointswach_temporal_corr',
        '--debug_file', f'{s2shores_paths.debug_dir}/debug_points_SWASH_7_4_cropped.yaml'])
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/debug_pointswach_temporal_corr_quick_5s",
                  output_dir=s2shores_paths.output_dir)


def test_debug_pointswach_spatial_dft_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test SWASH8.2 data without ROI, with geotiff product, spatial dft debug and grid debug point mode.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    unzip_file(s2shores_paths.swach8_cropped.with_suffix('.zip'))
    runner = CliRunner()

    runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.swach8_cropped),
        '--product_type', 'geotiff',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config5/wave_bathy_inversion_config_quick.yaml',
        '--debug_path', f'{s2shores_paths.output_dir}/debug_pointswach_spatial_dft',
        '--debug_file', f'{s2shores_paths.debug_dir}/debug_points_SWASH_8_2_cropped.yaml'])
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/debug_pointswach_dft_quick_108s",
                  output_dir=s2shores_paths.output_dir)


def test_debug_pointswach_spatial_corr_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test SWASH8.2 data without ROI, with geotiff product, spatial
    correlation debug, grid debug point mode and Layers-type nominal.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    unzip_file(s2shores_paths.swach8_cropped.with_suffix('.zip'))
    runner = CliRunner()

    runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.swach8_cropped),
        '--product_type', 'geotiff',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config6/wave_bathy_inversion_config_quick.yaml',
        '--debug_path', f'{s2shores_paths.output_dir}/debug_pointswach_spatial_corr',
        '--debug_file', f'{s2shores_paths.debug_dir}/debug_points_SWASH_8_2_cropped.yaml'])
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/debug_pointswach_spatial_corr_quick_114s",
                  output_dir=s2shores_paths.output_dir)


def test_debug_area_funwave_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Funwave data with geotiff product and debug area.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    unzip_file(s2shores_paths.funwave_cropped.with_suffix('.zip'))
    runner = CliRunner()

    runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.funwave_cropped),
        '--product_type', 'geotiff',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config9/wave_bathy_inversion_config_quick.yaml',
        '--debug_path', f'{s2shores_paths.output_dir}/debug_area_funwave',
        '--debug_file', f'{s2shores_paths.debug_dir}/debug_area_funwave_cropped.yaml'])
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/debug_area_funwave_quick_35s",
                   output_dir=s2shores_paths.output_dir)


