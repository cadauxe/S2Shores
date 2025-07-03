
# S2Shores

Welcome to S2Shores, a Python package for estimating wave characteristics to derive bathymetries.
S2Shores is a Python package designed to estimate wave characteristics for deriving bathymetries, initially tailored for optical spaceborne data but also compatible with other sensors like RADAR or LiDAR, given adequate spatiotemporal sampling. The package aims to detect bulk wave displacement over a time delay and estimate two of the five key variables (c, T, L, w, k) to solve for bathymetry using linear dispersion. It implements three methods: spatial DFT, spatial correlation, and temporal correlation. The spatial DFT method, designed for Sentinel 2 imagery, uses two images with a small time delay to find wave directions and characteristics through a Radon Transform and FFT. The spatial correlation method is similar but starts with a 2D image correlation. The temporal correlation method, robust to wave breaking conditions, uses random point sampling and pair-wise time-series correlation, applicable to various spatially distributed time-series data.

- **Spatial DFT**: Uses two images/frames with a small time delay to find multiple wave directions and characteristics through a Radon Transform, FFT, and cross-spectral correlation.
- **Spatial Correlation**: Similar to spatial DFT but starts with a 2D image correlation and applies a Radon Transform.
- **Temporal Correlation**: A time-series correlation method robust to wave breaking conditions, applicable to various types of spatially distributed time-series.

S2Shores is coded in an object-based fashion, with classes separated by specific tasks or specialisms to enable efficient large-scale computing. A global orchestrator ("global_bathymetry") drives local bathymetry estimation using specific methods within the "local_estimator" class. The code modularly handles image processing, wave detection, and wave physics. To modify a method, start with the "local_estimator" class, create a separate branch using git, and commit changes after successful testing. Automatic non-regression tests ensure the quality and precision of S2Shores, with contributions accepted only after passing these tests. The package philosophy emphasizes clean, minimally repetitive code with minimal dependencies, preferring low-level packages like GDAL over umbrella packages like Rasterio. The output is a GIS-ready NetCDF file, easily importable and visualizable in QGIS or xarray in a Python environment.

As a matter of workflow, we recommend reviewing input images to confirm observable wave displacement, as satellite imagery and wave-signal visibility depend on various factors, such as sun angle and season. Lower the output resolution and activate debug mode with plotting to understand S2Shores' process step-by-step for a few points initially and/or have a look at the provided notebooks per method in the /notebooks section. The documentation is a work in progress, and contributions to both code and documentation are encouraged.

The work of S2Shores is an effort of a small group of people, we welcome you to actively contribute to this package to commonly move forward. This is why the chosen license is apache 2.0, which allow you to use the package without contamination in your workflow or usage. The pages, readthedocs for example, are not (yet) perfect or exhaustive, it is merely a work in progress. Slowly but surely we will update the pages, and we invite you when you contribute to the code, to also write a section on the contribution in the manual and read-the-docs. Notebooks are added to provide a clear entry into the code, and perform point analysis per method. When contributing, a small notebook, explaining a modification or new method, would be extremely appreciated. 

Ok, that’s it for the introduction. Enjoy and have fun!

# Online documentation

The online documentation can be found [here](https://s2shores.readthedocs.io/en/latest/).
<!-- Change link if necessary when final documentation has been pushed -->

# Run environment

S2Shores is python based, and as most python projects we prefer to work in an S2Shores specific python-environment. We presume, that if you read through the introduction, and you manage to come here that you are more than capable to install anaconda, miniconda or directly python on your PC or MAC. Once one of these are installed one can choose one of the two procedures to create an environment with S2Shores (pip or conda), for Windows, or Linux (and for an installation on MAC, follow Linux).

For a detailed description we refer to the [installation documentation](https://s2shores.readthedocs.io/en/latest/install.html).
<!-- Change link if necessary when final documentation has been pushed -->

# How to run s2shores

After installation, S2Shores can simply be ran in the command-line by using the command : ``s2shores`` + input. There is obligatory commands (input_product, product_type, output_dir and config_file) and optional inputs denoted between squared brackets “[ ]” below.


In more detail, the command ``s2shores`` takes the following arguments :

``--input_product`` Path to the input product. See below for further information (**Products** section).

``--product_type`` Type of the input product. Choice between S2 and geotiff.

``--output_dir`` The directory where the output results of s2shores will be stored.

``--config_file`` YAML configuration file for bathymetry computation (wave_bathy_inversion_config.yaml).

``[--debug_file]`` YAML file defining points or area to spy for debug purpose. Example of debug files are given [here](https://github.com/CNES/S2Shores/tree/main/tests/data/debug).
<!-- Change link when branch has been merged -->

``[--debug_path]`` Path to store debug information.

``[--distoshore_file]`` Georeferenced netCDF file or GeoTif file giving the distance of a point to the closest shore. This information is used to compute bathymetry only on the sea. If not specified, bathymetry is computed over the complete image footprint.

``[--delta_times_dir]`` Directory containing the files describing S2A, S2B and S2C delta times between detectors. Mandatory for processing a Sentinel2 product. Example of delta_times files for S2A, S2B and S2C based and the ESA handbook (delta_t constant per band) and CNES corrected delta-times are given [here](https://github.com/CNES/S2Shores/tree/main/src/s2shores/bathylauncher/config). 
<!-- Change link when branch has been merged -->

``[--roi_file]`` Vector file specifying the polygon(s) where the bathymetry must be computed (geojson file format for instance). 

``[--limit_to_roi]`` If set and roi_file is specified, limit the bathymetry output to that roi.

``[--nb_subtiles]`` 1 by default. The input product scene is divided into subtiles that can be processed independently.

``[--sequential]`` If set, allows run in a single thread, useful for debugging purpose.

``[--profiling]`` If set, print profiling information about the whole bathymetry estimation.


Detailed information on the configuration can be found here :
- *wave_bathy_inversion_config.yaml* (an example can be found in the s2shores [config directory](https://github.com/CNES/S2Shores/blob/main/tests/data/config/config2/wave_bathy_inversion_config.yaml)) : parameters for the bathymetry inversion method.



One configuration file is needed :
- *wave_bathy_inversion_config.yaml* (an example can be found in the s2shores [config directory](https://github.com/CNES/S2Shores/blob/main/tests/data/config/config2/wave_bathy_inversion_config.yaml)) : parameters for the bathymetry inversion method.


# Main parameters

 #### *wave_bathy_inversion_config.yaml*

 - WAVE_EST_METHOD: choice b.w. 3 estimation methods (SPATIAL_DFT and SPATIAL_CORRELATION recommended for S2 products, TEMPORAL_CORRELATION for video sequence).
 - SELECTED_FRAMES: list of frames to be used from the input product to perform the bathymetry estimation. For S2 products, it corresponds to S2 bands, they should be of the same resolution (example : "B02" "B04"). If empty list, all available frames in the product will be selected.
 - DXP, DYP : resolution of the bathymetry product.
 - WINDOW : size of the window used to compute the waves characteristic in one point.
 - NKEEP : number of main waves trains to consider. Depth information is computed for each wave train (available only with the SPATIAL_DFT method).
 - LAYERS_TYPE : DEBUG, EXPERT or NOMINAL. In NOMINAL mode the bathymetry product contains only the following
layers : Status, Depth, Direction, Wavelength and Celerity. In EXPERT mode, more layers may be provided, some of them depending on the estimation
method: Gravity, Distoshore, Period, Wavenumber, Delta Celerity, Phase Shift, Delta Acquisition Time, Waves Linearity, Period Offshore, Travelled Distance. In DEBUG mode, additional layers specific to the estimation method are also provided: Energy, Delta Phase Ratio, Energy Ratio for the Spatial DFT estimation method.
 - OUTPUT_FORMAT : GRID (by default) or POINT. In the default mode, the bathymetry product is given as a mapping grid respecting the specified resolutions. In this mode, debug points have to be points of the grid. In the "POINT mode", it is possible to give (in a debug_file) a list of points, not grid-constrained. The resulting bathy product contains the corresponding list of bathymetry results.


# Launch() API

It is also possible to launch a bathymetry estimation by using the launch() function. 

#### Arguments :

``products: ProductsDescriptor`` a dictionary of input products. For each product, the following characteristics are specified :
 
    Path,              # Path to the product, either a file or a directory
    Type[OrthoStack],  # Type to use for accessing the product (GeoTiffProduct or S2ImageProduct)
    Path,              # Path to a directory where bathymetry will be written
    dict,              # A dictionary containing the processing parameters (from wave_bathy_inversion_config.yaml)
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

``gravity_type: Optional[str]`` None by default (CONSTANT). Specify which Gravity Provider to use, either CONSTANT or LATITUDE_VARYING.

``cluster: Optional[SpecCluster]`` None by default (local cluster). Specify a cluster to be used by the dask dataframe (a PBScluster for  instance).

``sequential_run: bool`` False by default. Set to True to allow profiling and debugging by running in a single thread.

# Products

#### Geotiff :

``--input_product path_to/MyFile.tif``

The geotiff input type is used to compute bathymetry on a sequence of superimposable frames. The geotiff image contains all the frames sorted in a chronological order (one band by frame).

A json file is associated with the geotiff file to provide some complementary data (the geotiff and the json files should have the same name and be located in the same directory).

Example of json file for a product containing 5 frames :

    {"SATELLITE":"MySat",
     "ACQUISITION_TIME":"20220614T113447",
     "FRAMES_TIME":
        {"1":"20220614T11:34:01.264000+00:00"
         "2":"20220614T11:34:03.751000+00:00"
         "3":"20220614T11:34:05.325000+00:00"
         "4":"20220614T11:34:07.256000+00:00"
         "5":"20220614T11:34:09.568000+00:00"
        },
     "PROCESSING_LEVEL":"Product level of the input images",
     "ZONE_ID":"MyZone"
    }

FRAMES_TIME is used to compute the exact temporal delay between two frames.
The other data will be given as informations in the bathymetry product.

#### S2:

``--input_product path_to/S2*_MSIL1C_*_*_*_T*_*.SAFE``

Bathymetry is computed on Sentinel2 L1C products (ESA format). 


# References
In the case of the use of recalculated CNES time-lag values using Sentinel 2 for dynamic problemsets, please cite : 
Binet, R., Bergsma, E., and Poulain, V. (2022) ACCURATE SENTINEL-2 INTER-BAND TIME DELAYS, ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci., V-1-2022, 57–66, https://doi.org/10.5194/isprs-annals-V-1-2022-57-2022 

Almar, R., Bergsma, E. W., Maisongrande, P., & De Almeida, L. P. M. (2019). Wave-derived coastal bathymetry from satellite video imagery: A showcase with Pleiades persistent mode. Remote Sensing of Environment, 231, 111263. https://doi.org/10.1016/j.rse.2019.111263

Almar, R.; Bergsma, E.W.J.; Brodie, K.L.; Bak, A.S.; Artigues, S.; Lemai-Chenevier, S.; Cesbron, G.; Delvit, J.-M. (2022 )Coastal Topo-Bathymetry from a Single-Pass Satellite Video: Insights in Space-Videos for Coastal Monitoring at Duck Beach (NC, USA). Remote Sens. 14, 1529. https://doi.org/10.3390/rs14071529

Almar, R.,  Bergsma, E.W.J., Thoumyre, G., Lemai-Chenevier, S., Loyer, S. Artigues, S., Salles, G., Garlan, T., Lifermann, A. (2024) Satellite-derived bathymetry from correlation of Sentinel-2 spectral bands to derive wave kinematics: Qualification of Sentinel-2 S2Shores estimates with hydrographic standards, Coastal Engineering, Volume 189,2024,104458,ISSN 0378-3839, https://doi.org/10.1016/j.coastaleng.2024.104458.

Bergsma, E.W.J.; Almar, R.; Maisongrande, P. (2019). Radon-Augmented Sentinel-2 Satellite Imagery to Derive Wave-Patterns and Regional Bathymetry. Remote Sens. , 11, 1918. https://doi.org/10.3390/rs11161918

Bergsma, E.W.J., Almar, R., Rolland, A., Binet, R., Brodie, K. L., & Bak, A. S. (2021). Coastal morphology from space: A showcase of monitoring the topography-bathymetry continuum. Remote Sensing of Environment, 261, 112469. https://doi.org/10.1016/j.rse.2019.111263 
