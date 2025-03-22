.. _api:

======================
Python API
======================

This section describes the API of **S2shores**.

------------
API Function
------------

It is possible to launch a bathymetry estimation by using the `launch() <https://github.com/cadauxe/S2Shores/blob/industrialisation/src/s2shores/bathylauncher/bathy_launcher.py>`_ function.

**Arguments :**

* ``products: ProductsDescriptor`` : A dictionary of input products. For each product, the following characteristics are specified in a tuple object :

    .. code-block:: python

        ProductsDescriptor = Dict[
            str,                     # Name of the product
            Tuple[
                  Path,              # Path to the product, either a file or a directory (see the [Products]_ section for more information)
                  Type[OrthoStack],  # Type to use for accessing the product (GeoTiffProduct or S2ImageProduct)
                  Path,              # Path to a directory where bathymetry will be written
                  dict,              # A dictionary containing the processing parameters of wave_bathy_inversion_config.yaml (see the [Configuration file]_ section for more information)
                  int,               # Maximum number of subtiles to process
                  Optional[Path],    # Path to a file or a directory containing specific data to be used
                                     # by the DeltaTimeProvider associated to the product.
                  Optional[Path],    # Path to a file or a directory containing specific data to be used
                                     # by the DisToShoreProvider associated to the product.
                  Optional[Path],    # Path to a geojson or shapefile defining a ROI
                  bool,              # If True, the produced bathymetry will be limited to a bounding
                                     # box enclosing the Roi with some margins.
                  Optional[dict],    # A dictionary containing the points or areas to debug
                  Optional[Path]     # Path to send debugging results
            ]
        ]

    See the `Products`_ and `Configuration file`_ sections for more information about the products and parameter files to provide.

* ``gravity_type: Optional[str]`` : None by default (CONSTANT). Specify which Gravity Provider to use, either CONSTANT or LATITUDE_VARYING.

* ``cluster: Optional[SpecCluster]`` : None by default (local cluster). Specify a cluster to be used by the dask dataframe (a PBScluster for  instance).

* ``sequential_run: bool`` : False by default. Set to True to allow profiling and debugging by running in a single thread.


--------
Products
--------

Two types of input products can be provided, either in Geotiff format or in Sentinel-2 SAFE format.

* **Geotiff**

    *Example of argument :* ``path_to/MyFile.tif``

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

    *Example of argument :* ``path_to/S2*_MSIL1C_*_*_*_T*_*.SAFE``

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

