# -*- coding: utf-8 -*-
""" Definition of the BathyLauncher class and associated functions

:author: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2024 CNES. All rights reserved.
:license: see LICENSE file
:created: 24 November 2021

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
import logging
import warnings
from pathlib import Path

from typing import Type, Dict, Tuple, Optional, List  # @NoMove

import dask
from dask.distributed import Client, LocalCluster, SpecCluster

from shapely.geometry import Point

from dask import delayed, dataframe  # @NoMove

import s2shores
from s2shores.global_bathymetry.bathy_estimator import BathyEstimator
from s2shores.image.ortho_stack import OrthoStack
from s2shores.waves_exceptions import ProductNotFound

ProductsDescriptor = Dict[
    str,
    Tuple[Path,              # Path to the product, either a file or a directory
          Type[OrthoStack],  # Type to use for accessing the product
          Path,              # Path to a directory where bathymetry will be written
          dict,              # A dictionary containing the processing parameters
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


LOGGER = logging.getLogger('bathy')


class BathyLauncher:
    """ A BathyLauncher allows to run the computation of bathymetry on several products.
    The products may be of different types.
    The bathymetry parameters may vary from one product to the other.
    It supports running on a dask cluster.
    """

    def __init__(self,
                 cluster: Optional[SpecCluster] = None, sequential_run: bool = False) -> None:
        self.estimators: List[BathyEstimator] = []
        if cluster is None:
            cluster = LocalCluster()
            preferred_scheduler = 'processes'
        else:
            preferred_scheduler = 'distributed'
        self._client = Client(cluster)
        self._scheduler = 'single-threaded' if sequential_run else preferred_scheduler
        dask.config.set(scheduler=self._scheduler)

    @classmethod
    def launch(cls, products: ProductsDescriptor,
               gravity_type: Optional[str] = None,
               cluster: Optional[SpecCluster] = None,
               sequential_run: bool = False,
               tries_max: int = 100) -> None:
        """ Creates a BathyLauncher, populates it with some products to process and launch the
        bathymetry estimation on each of them using a cluster.

        :param products: the products for which bathy estimation must be computed. Each entry in
                         this dictionary provides the parameters describing the product and its
                         processing parameters.
        :param gravity_type: 'CONSTANT' or 'LATITUDE_VARYING' to choose one gravity model
        :param cluster: a specific dask cluster to be used for the run. When unspecified a default
                        local cluster is used.
        :param sequential_run: when True allows profiling and debugging by running in a single
                               thread
        :param tries_max: Maximum try number before stop. Used to handle heavy amount of
                          simultaneous files opening (try/catch/wait/retry)
        """
        bathy_launcher = cls(cluster, sequential_run=sequential_run)
        for product_name, (product_path, product_cls, output_path, wave_params, nb_subtiles,
                           delta_times_path, distoshore_file_path, roi_file_path, limit_to_roi,
                           debug_params, debug_path) in products.items():

            i = 0
            while i<tries_max:
                i += 1
                try:

                    # Set chains versions if the key is missing
                    if 'CHAINS_VERSIONS' not in wave_params.keys():
                        wave_params['CHAINS_VERSIONS'] = \
                            f's2shores : {s2shores.__version__}'

                    # Add product and get its bathy estimator
                    outputmode = wave_params['GLOBAL_ESTIMATOR']['OUTPUT_FORMAT']
                    if outputmode == 'POINT' and nb_subtiles>1:
                        nb_subtiles = 1
                        warnings.warn("The use of the POINT OUTPUT_FORMAT is only compatible with "
                                      "NB_SUBTILES set to 1. Nb_subtiles has been set to 1.")			        
                    estimator = bathy_launcher.add_product(product_name, product_path, product_cls,
                                                        output_path, wave_params, nb_subtiles)

                    # Set the gravity provider.
                    estimator.set_gravity_provider(provider_info=gravity_type)

                    # Set the distoshore provider
                    estimator.set_distoshore_provider(provider_info=distoshore_file_path)

                    # Set the delta time provider.
                    estimator.set_delta_time_provider(provider_info=delta_times_path)

                    # Set the Roi provider.
                    estimator.set_roi_provider(provider_info=roi_file_path,
                                               limit_to_roi=limit_to_roi)

                    # Create subtiles (mandatory for setting debug area)
                    estimator.create_subtiles()

                    # Set points to debug
                    if debug_params is not None:
                        debug_points = debug_params.get('DEBUG_POINTS')
                        debug_area = debug_params.get('DEBUG_AREA')
                        if debug_points is not None:
                            estimator.set_debug_samples([Point(float(point[0]), float(point[1]))
                                                         for point in debug_points])
                        elif estimator.output_format == 'POINT':
                            raise SystemExit(
                                'User must give a list of points if OUTPUT_FORMAT is POINT.')

                        if debug_area is not None:
                            estimator.set_debug_area(Point(debug_area['BOTTOM_LEFT_CORNER']),
                                                     Point(debug_area['TOP_RIGHT_CORNER']),
                                                     debug_area['DECIMATION_NUMBER'])
                        estimator.debug_path = debug_path
                    elif estimator.output_format == 'POINT':
                        raise SystemExit('User must give a list of points if '
                                         'OUTPUT_FORMAT is POINT.')
                    break
                except OSError as e:
                    if e.errno==24:
                        LOGGER.error(f'OSError : too many open files : {e}')
                        # OSError [24] is : Too many open files
                        time.sleep(5) # wait for some processes to finish reading
                        continue      # then try again
                    else:
                        raise(OSError(e)) # crashing for another unknown OSError

        if i==tries_max:
            # Stop trying after too many tries (there should be another solution)
            LOGGER.error(f'Too many open files ({tries_max} tries), '
                         f'try to manually set ulimit higher.')
        bathy_launcher.run()

    def add_product(self, product_name: str,
                    product_path: Path, product_cls: Type[OrthoStack], output_path: Path,
                    wave_params: dict, nb_subtiles: int) -> BathyEstimator:
        """ Add a product to the set of products to be processed by this BathyLauncher.

        :param product_name: the product name used for logging info
        :param product_path: the path to file or directory corresponding to this product.
        :param product_cls: the type of this product, a class inheriting from OrthoStack
        :param output_path: path to the directory where the netCDF bathy file will be written.
        :param wave_params: the set of parameters to be used by the bathymetry inversion algorithms
        :param nb_subtiles: the number of subtiles in which the product must be split for speeding
                            up the processing.
        :returns: the bathymetry estimator which will handle the bathymetry inversion for this
                  product.
        :raises ProductNotFound: when the product cannot be found at the provided path.
        """
        product_id = f'product: {product_name} ({product_cls.__name__})'
        if not product_path.exists():
            log_msg = f"{product_id} doesn't exist at the given path: {product_path}."
            log_msg += ' Skipping bathymetry estimation.'
            LOGGER.error(log_msg)
            raise ProductNotFound(log_msg)
        log_msg = f'Processing {product_id}'
        LOGGER.info(log_msg)

        product = product_cls(product_path)

        # Create the BathyEstimator object
        estimator = BathyEstimator(product, wave_params, output_path, nb_subtiles)
        self.estimators.append(estimator)

        return estimator

    def run(self) -> None:
        """ run the BathyLauncher on all the products which has been added to it.

        """
        # Prepare the running graph of the BathyEstimator on all subtiles of one product
        bathy_products = []
        for estimator in self.estimators:
            bathy_subtiles = [delayed(estimator.compute_bathy_for_subtile)(i)
                              for i in range(estimator.nb_subtiles)]
            bathy_product = delayed(estimator.merge_subtiles)(bathy_subtiles)
            bathy_products.append(bathy_product)
        # Run the whole graph on all products - fingers crossed!
        dataframe.compute(*bathy_products)
