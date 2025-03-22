.. _cli:

======================
Command Line Interface
======================

This section describes the CLI of **S2shores**.

----------------
S2Shores command
----------------

The CLI command of **S2shores** is the following :

.. code-block:: console

    $ s2shores --help
    Usage: s2shores [OPTIONS]

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


--------
Products
--------

Two types of input products can be provided, either in Geotiff format or in Sentinel-2 SAFE format.

* **Geotiff**

    *Example :* ``--input_product path_to/MyFile.tif``

    The geotiff input type is used to compute bathymetry on a sequence of super-imposable frames. The geotiff image contains all the frames sorted in a chronological order (one band by frame).

    A json file is associated with the geotiff file to provide some complementary data (the geotiff and the json files should have the same name and be located in the same directory).

    Example of json file for a product containing 5 frames :

    .. code-block:: json

        {
         "SATELLITE":"MySat",
         "ACQUISITION_TIME":"20220614T113447",
         "FRAMES_TIME":
            {
             "1":"20220614T11:34:01.264000+00:00"
             "2":"20220614T11:34:03.751000+00:00"
             "3":"20220614T11:34:05.325000+00:00"
             "4":"20220614T11:34:07.256000+00:00"
             "5":"20220614T11:34:09.568000+00:00"
            },
         "PROCESSING_LEVEL":"Product level of the input images",
         "ZONE_ID":"MyZone"
        }

    *FRAMES_TIME* is used to compute the exact temporal delay between two frames.
    The other data will be given as informations in the bathymetry product.



* **Sentinel-2 SAFE**

    *Example :* ``--input_product path_to/S2*_MSIL1C_*_*_*_T*_*.SAFE``

    Bathymetry is computed on Sentinel2 L1C products (PEPS format).


------------------
Configuration file
------------------

One configuration file is needed to provide the parameters for the bathymetry inversion method.
It must be named *wave_bathy_inversion_config.yaml* (an example can be found in the `S2shores GitHub <https://github.com/CNES/S2Shores/blob/main/config/wave_bathy_inversion_config.yaml>`_.

The main parameters in *wave_bathy_inversion_config.yaml* are :

 - **WAVE_EST_METHOD** : Choice b.w. 3 estimation methods (SPATIAL_DFT and SPATIAL_CORRELATION recommended for S2 products, TEMPORAL_CORRELATION for video sequence).
 - **SELECTED_FRAMES** : List of frames to be used from the input product to perform the bathymetry estimation. For S2 products, it corresponds to S2 bands, they should be of the same resolution (example : "B02" "B04"). If empty list, all available frames in the product will be selected.
 - **DXP, DYP** : Resolution of the bathymetry product.
 - **WINDOW** : Size of the window used to compute the waves characteristic in one point.
 - **NKEEP** : Number of main waves trains to consider. Depth information is computed for each wave train (available only with the SPATIAL_DFT method).
 - **LAYERS_TYPE** : DEBUG, EXPERT or NOMINAL.
    In NOMINAL mode the bathymetry product contains only the following :

    - layers : Status, Depth, Direction, Wavelength and Celerity. In EXPERT mode, more layers may be provided, some of them depending on the estimation
    - method: Gravity, Distoshore, Period, Wavenumber, Delta Celerity, Phase Shift, Delta Acquisition Time, Waves Linearity, Period Offshore, Travelled Distance. In DEBUG mode, additional layers specific to the estimation method are also provided: Energy, Delta Phase Ratio, Energy Ratio for the Spatial DFT estimation method.
 - **OUTPUT_FORMAT** : GRID (by default) or POINT. In the default mode, the bathymetry product is given as a mapping grid respecting the specified resolutions. In this mode, debug points have to be points of the grid. In the "POINT mode", it is possible to give (in a debug_file) a list of points, not grid-constrained. The resulting bathy product contains the corresponding list of bathymetry results.


