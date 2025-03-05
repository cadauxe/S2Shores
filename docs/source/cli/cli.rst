.. _tutorial:

========
Tutorial
========


S2shores can be used either through its python interface, or within its CLI.

### Command Line Interface

The cli command is the following :
Usage: bathy_processing.py [OPTIONS]

Options:
  --input_product PATH            Path to input product  [required]
  --product_type [S2|geotiff]     [required]
  --output_dir PATH               Output directory.  [required]
  --config_file PATH              YAML config file for bathymetry computation
                                  [required]
  --debug_file PATH               YAML config file for bathymetry debug
                                  definition
  --debug_path PATH               path to store debug information
  --distoshore_file PATH          georeferenced netCDF file giving the
                                  distance of a point to the closest shore
  --delta_times_dir PATH          Directory containing the files describing
                                  S2A and S2B delta times between detectors.
                                  Mandatory for processing a Sentinel2
                                  product.
  --roi_file PATH                 vector file specifying the polygon(s) where
                                  the bathymetry must be computed
  --limit_to_roi                  if set and roi_file is specified, limit the
                                  bathymetry output to that roi
  --nb_subtiles INTEGER           Number of subtiles
  --sequential / --no-sequential  if set, allows run in a single thread,
                                  usefull for debugging purpose
  --profiling / --no-profiling    If set, print profiling information about
                                  the whole bathymetry estimation
  --help                          Show this message and exit.


Output directory must exists before computing the bathymetry.

More about input product

....

### Python API

estimator = bathy_launcher.add_product(product_name, product_path, product_cls,
                                                        output_path, wave_params, nb_subtiles)

# Set the gravity provider.
estimator.set_gravity_provider(provider_info=gravity_type)


# Set the distoshore provider
estimator.set_distoshore_provider(provider_info=distoshore_file_path)

# Set the delta time provider.
estimator.set_delta_time_provider(provider_info=delta_times_path)

# Set the Roi provider.
estimator.set_roi_provider(provider_info=roi_file_path, limit_to_roi=limit_to_roi)

# Create subtiles (mandatory for setting debug area)
estimator.create_subtiles()


Creating plots



