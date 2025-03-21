# S2Shores

======================

This project gathers several estimators to be able to compute bathymetry from standard format such as sentinel2 or geotiff.

It uses methods based on the inversion of wave peaks and data provider services (delta time b.w. frames, gravity depending on latitude, distance to shore) for bathymetry estimation.

The online documentation can be found [here](https://s2shores.readthedocs.io/en/latest/).
<!-- Change link if necessary when final documentation has been pushed -->

# Environment

To create an environment with S2shores, two procedures are available with pip or conda.

Please refer to the [installation documentation](https://s2shores.readthedocs.io/en/latest/install.html).
<!-- Change link if necessary when final documentation has been pushed -->


# Context

One configuration file is needed :
- *wave_bathy_inversion_config.yaml* (an example can be found in the s2shores [config directory](https://github.com/CNES/S2Shores/blob/main/config/wave_bathy_inversion_config.yaml)) : parameters for the bathymetry inversion method.


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

# Command

A command is available : ``bathy_processing.py`` 

It takes the following arguments :

``--input_product`` Path to the input product. See below for further information (**Products** section).

``--product_type`` Type of the input product. Choice between S2 and geotiff.

``--output_dir``

``--config_file`` YAML configuration file for bathymetry computation (wave_bathy_inversion_config.yaml).

``[--debug_file]`` YAML file defining points or area to spy for debug purpose. Example of debug files are given [here](https://github.com/CNES/S2Shores/tree/main/tests/data/debug).
<!-- Change link when branch has been merged -->

``[--debug_path]`` Path to store debug information.

``[--distoshore_file]`` Georeferenced netCDF file or GeoTif file giving the distance of a point to the closest shore. This information is used to compute bathymetry only on the sea. If not specified, bathymetry is computed over the complete image footprint.

``[--delta_times_dir]`` Directory containing the files describing S2A and S2B delta times between detectors. Mandatory for processing a Sentinel2 product. Example of delta_times files for S2A and S2B based and the ESA handbook (delta_t constant per band) are given [here](https://github.com/CNES/S2Shores/tree/main/src/s2shores/bathylauncher/config). 
<!-- Change link when branch has been merged -->

``[--roi_file]`` Vector file specifying the polygon(s) where the bathymetry must be computed (geojson file format for instance). 

``[--limit_to_roi]`` If set and roi_file is specified, limit the bathymetry output to that roi.

``[--nb_subtiles]`` 1 by default. The input product scene is divided into subtiles that can be processed independently.

``[--sequential]`` If set, allows run in a single thread, usefull for debugging purpose.

``[--profiling]`` If set, print profiling information about the whole bathymetry estimation.



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

Bathymetry is computed on Sentinel2 L1C products (PEPS format). 



# References

Almar, R.; Bergsma, E.W.J.; Brodie, K.L.; Bak, A.S.; Artigues, S.; Lemai-Chenevier, S.; Cesbron, G.; Delvit, J.-M. Coastal Topo-Bathymetry from a Single-Pass Satellite Video: Insights in Space-Videos for Coastal Monitoring at Duck Beach (NC, USA). Remote Sens. 2022, 14, 1529. https://doi.org/10.3390/rs14071529

Bergsma, E.W.J.; Almar, R.; Maisongrande, P. (2019). Radon-Augmented Sentinel-2 Satellite Imagery to Derive Wave-Patterns and Regional Bathymetry. Remote Sens. , 11, 1918. https://doi.org/10.3390/rs11161918

Bergsma, E.W.J., Almar, R., Rolland, A., Binet, R., Brodie, K. L., & Bak, A. S. (2021). Coastal morphology from space: A showcase of monitoring the topography-bathymetry continuum. Remote Sensing of Environment, 261, 112469. https://doi.org/10.1016/j.rse.2019.111263 


