import glob
import zipfile
import xarray as xr
import os


def compare_files(reference_dir : str, output_dir : str, debug_dir : str = None):
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

    ref_files = sorted(os.listdir(reference_dir))
    out_test_files = sorted(os.listdir(out_test_dir))

    if "debug" in ref_files :
        ref_files.remove("debug")
        assert debug_dir != None
        ref_debug_dir = os.path.join(reference_dir, "debug")
        ref_debug = sorted(os.listdir(ref_debug_dir))
        out_test_debug = sorted(os.listdir(debug_dir))

        if ref_debug == out_test_debug:
            print("Both directories contain the same filenames.")
        else:
            raise Exception("Debug files differ between the directories.\n"
                            f"Only in {ref_debug_dir} : {[item for item in ref_debug if item not in out_test_debug]}\n"
                            f"Only in {debug_dir} : {[item for item in out_test_debug if item not in ref_debug]}")

    if ref_files == out_test_files:
        print("Both directories contain the same filenames.")
    else:
        raise Exception("Filenames differ between the directories.\n"
                            f"Only in {reference_dir} : {[item for item in ref_files if item not in out_test_files]}\n"
                            f"Only in {out_test_dir} : {[item for item in out_test_files if item not in ref_files]}")

    #Assert the files in the reference directory are the same
    #than the ones in the lastly created directory
    compare_nc = False
    for nc_file in ref_files:
        if ".nc" in nc_file:
            ref_nc = nc_file
            compare_nc = True
    for nc_file in out_test_files:
        if ".nc" in nc_file:
            out_nc = nc_file

    if compare_nc:
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