# -*- coding: utf-8 -*-
""" Class performing bathymetry computation using temporal correlation method

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 18/06/2021

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""

import os
from typing import TYPE_CHECKING, Optional

import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..image.image_geometry_types import PointType
from ..image.ortho_sequence import OrthoSequence
from ..local_bathymetry.temporal_correlation_bathy_estimator import (
    TemporalCorrelationBathyEstimator)
from ..waves_exceptions import (CorrelationComputationError, NotExploitableSinogram,
                                WavesEstimationError)
from .local_bathy_estimator_debug import LocalBathyEstimatorDebug

if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport


class TemporalCorrelationBathyEstimatorDebug(LocalBathyEstimatorDebug,
                                             TemporalCorrelationBathyEstimator):
    """ Class performing debugging for temporal correlation method
    """

    def __init__(self, location: PointType, ortho_sequence: OrthoSequence,
                 global_estimator: 'BathyEstimator',
                 selected_directions: Optional[np.ndarray] = None) -> None:
        # FIXME: Handle severals wave_estimations
        ######################################################
        super().__init__(location, ortho_sequence, global_estimator, selected_directions)

        # Create figure
        self._figure = plt.figure(constrained_layout=False, figsize=[9, 14])
        self._gs = gridspec.GridSpec(5, 2, figure=self._figure, wspace=0.7, hspace=0.5)

        # Set Main title
        plt.suptitle(f'Debug info \n Point {self.location.x} {self.location.y}',
                     fontsize='x-large',
                     bbox=dict(boxstyle='round', facecolor='lightgrey', edgecolor='red')
                     )

    def run(self) -> None:
        try:
            super().run()

        except WavesEstimationError as excp:
            self.explore_results()
            raise excp

        except NotExploitableSinogram as excp:
            self.show_first_frame()
            self.show_first_frame_selection()
            self.show_first_TS()
            self.show_correlation_matrix()
            self.show_correlation_matrix_filled_filtered()
            self.show_radon_matrix()
            self.show_failed_sinogram()
            self.show_depth_esti_values()
            self.dump_figure()
            raise excp

        except CorrelationComputationError as excp:
            self.show_first_frame()
            self.show_first_frame_selection()
            self.show_first_TS()
            self.show_depth_esti_values()
            self.dump_figure()
            raise excp

    def show_first_frame(self) -> None:
        """ Show first frame in sequence for a debug point
        """
        # Import 1st frame
        first_image = self.ortho_sequence[0].pixels
        imin = np.min(first_image)
        imax = np.max(first_image)

        # Retrieve window spatial shape in meters
        spatial_res = self.metrics['spatial_resolution']
        wind_shape = first_image.shape
        x_spatial_limits = np.array([0, wind_shape[1]]) * spatial_res
        y_spatial_limits = np.array([0, wind_shape[0]]) * spatial_res

        # Plot
        subfigure = self._figure.add_subplot(self._gs[0, 0])
        subfigure.imshow(
            first_image,
            norm=Normalize(
                vmin=imin,
                vmax=imax),
            extent=[
                x_spatial_limits[0],
                x_spatial_limits[1],
                y_spatial_limits[1],
                y_spatial_limits[0]])
        plt.title('1st frame')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')

        # Draw an arrow in the wave direction
        radius = min(x_spatial_limits[1], y_spatial_limits[1]) / 3
        if 'direction' in self.metrics:
            cartesian_dir_x = np.cos(np.deg2rad(self.metrics['direction']))
            cartesian_dir_y = -np.sin(np.deg2rad(self.metrics['direction']))
            subfigure.arrow(
                x_spatial_limits[1] // 2,
                y_spatial_limits[1] // 2,
                radius * cartesian_dir_x,
                radius * cartesian_dir_y)

    def show_first_frame_selection(self) -> None:
        """ Show selection of pixels used in the correlation for a debug point
        """
        # Import 1st frame
        first_image = self.ortho_sequence[0].pixels
        imin = np.min(first_image)
        imax = np.max(first_image)

        # Create the condition to select specific positions
        x_position = self._sampling_positions[0][0, :]
        y_position = self._sampling_positions[1][0, :]
        condition = np.zeros_like(first_image, dtype=bool)
        condition[x_position, y_position] = True

        # Plot in pixel positions
        subfigure = self._figure.add_subplot(self._gs[0, 1])
        subfigure.matshow(np.where(condition, first_image, np.nan), interpolation='none')
        plt.title('Selected pixels')
        plt.xlabel('Pixel ID X')
        plt.ylabel('Pixel ID Y')

    def show_first_TS(self) -> None:
        """ Show the first Normalised Timse-series of the dataset before and after BP filtering
        """
        subfigure = self._figure.add_subplot(self._gs[1, :])
        subfigure.plot(self.metrics['detrend_time_series'], label='Detrended TS')
        subfigure.plot(self.metrics['filtered_time_series'], label='Filtered TS')
        plt.title('1st random TS')
        plt.xlabel('Frame')
        plt.ylabel('NormIntensity')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)

    def show_correlation_matrix(self) -> None:
        """ Show correlation matrix where correlations were actually computed between selected pixels for a debug point
        """

        # Import correlation
        correlation_raw = self.metrics['projected_corr_raw']
        imin = np.min(self.correlation_image.pixels)
        imax = np.max(self.correlation_image.pixels)

        # Create the condition to select specific positions
        indices_x = self.metrics['corr_indices_x']
        indices_y = self.metrics['corr_indices_y']
        condition = np.zeros_like(correlation_raw, dtype=bool)
        condition[indices_x, indices_y] = True

        # Retrieve correlation spatial shape in meters
        spatial_res = self.metrics['spatial_resolution']
        wind_shape = self.ortho_sequence[0].pixels.shape
        x_spatial_limits = np.array([-(wind_shape[1]), wind_shape[1]]) * spatial_res
        y_spatial_limits = np.array([-(wind_shape[0]), wind_shape[0]]) * spatial_res

        # Plot
        subfigure = self._figure.add_subplot(self._gs[2, 0])
        pmc = subfigure.imshow(
            np.where(
                condition,
                correlation_raw,
                np.nan),
            extent=[
                x_spatial_limits[0],
                x_spatial_limits[1],
                y_spatial_limits[0],
                y_spatial_limits[1]])
        plt.title('Raw correlation')
        plt.xlabel('dX')
        plt.ylabel('dY')
        # create an axis for the colorbar
        axins = inset_axes(subfigure,
                           width='5%',
                           height='100%',
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=subfigure.transAxes,
                           borderpad=0
                           )
        plt.colorbar(pmc, cax=axins)

    def show_correlation_matrix_filled_filtered(self) -> None:
        """ Show correlation matrix with filled values filtered before the radon transform for a debug point
        """
        # Import correlation
        circular_corr = self.metrics['corr_radon_input']
        imin = np.min(circular_corr)
        imax = np.max(circular_corr)

        # Retrieve correlation spatial shape in meters
        spatial_res = self.metrics['spatial_resolution']
        wind_shape = self.ortho_sequence[0].pixels.shape
        x_spatial_limits = np.array([-(wind_shape[1]), wind_shape[1]]) * spatial_res * \
            self.local_estimator_params['TUNING']['RATIO_SIZE_CORRELATION']
        y_spatial_limits = np.array([-(wind_shape[0]), wind_shape[0]]) * spatial_res * \
            self.local_estimator_params['TUNING']['RATIO_SIZE_CORRELATION']

        # Plot
        subfigure = self._figure.add_subplot(self._gs[2, 1])
        pmc = subfigure.imshow(
            circular_corr,
            extent=[
                x_spatial_limits[0],
                x_spatial_limits[1],
                y_spatial_limits[0],
                y_spatial_limits[1]])
        plt.title('Filtered correlation')
        plt.xlabel('dX')
        plt.ylabel('dY')
        # create an axis for the colorbar
        axins = inset_axes(subfigure,
                           width='5%',
                           height='100%',
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=subfigure.transAxes,
                           borderpad=0
                           )
        plt.colorbar(pmc, cax=axins)

    def show_radon_matrix(self) -> None:
        """ Show sinogram for a debug point
        """
        # Import sinogram
        radon_array, _ = self.metrics['radon_transform'].get_as_arrays()

        # Retrieve rho axis of radon transform
        nb_rho, _ = radon_array.shape
        spatial_res = self.metrics['spatial_resolution']
        y_spatial_limits = np.array([-(nb_rho // 2), nb_rho // 2]) * spatial_res

        # Import directions
        directions = self.selected_directions
        min_dir = np.min(directions)
        max_dir = np.max(directions)
        ang_ticks = np.arange(min_dir, max_dir + 2, 45)
        ang_labels = ['{:.0f}'.format(ang) + u'\N{DEGREE SIGN}' for ang in ang_ticks]

        # Plot
        subfigure = self._figure.add_subplot(self._gs[3, :])
        subfigure.imshow(radon_array,
                         interpolation='nearest',
                         aspect='auto',
                         origin='lower',
                         extent=[min_dir, max_dir, y_spatial_limits[0], y_spatial_limits[1]]
                         )
        plt.plot(
            self.selected_directions,
            ((self._metrics['variances'] /
              np.max(
                self._metrics['variances'])) *
                (
                nb_rho -
                1) *
                spatial_res) +
            y_spatial_limits[0],
            'r')
        plt.title('Radon transform (sinogram)')
        plt.xticks(ticks=ang_ticks, labels=ang_labels)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.xlabel(r'$\theta$ (' + u'\N{DEGREE SIGN} from East)')
        plt.ylabel(r'$\rho$ (m)')

        # Highlight max var angle corresponding to wave direction
        if 'direction' in self.metrics:
            subfigure.arrow(
                self.metrics['direction'],
                y_spatial_limits[0],
                0,
                (nb_rho - 2) * spatial_res,
                color='orange')
            plt.annotate(f"{self.metrics['direction']}°",
                         (self.metrics['direction'] + 1, 10 + y_spatial_limits[0]),
                         color='orange',
                         fontweight='bold'
                         )

    def show_sinogram(self) -> None:
        """ Show sinogram values at a certain angle for a debug point
        """
        # Import sinogram
        sinogram_max_var = self.metrics['sinogram_max_var']

        # Retreive axis of the sinogram value plot
        spatial_res = self.metrics['spatial_resolution']
        x_spatial_axis = (np.arange(0, len(sinogram_max_var)) -
                          (len(sinogram_max_var) // 2)) * spatial_res
        min_limit_x = np.min(x_spatial_axis)
        min_limit_y = np.min(sinogram_max_var)

        # Import zeros and max detections
        zeros = self.metrics['wave_spatial_zeros']
        max_indices = self.metrics['max_indices']
        dx = self.metrics['wave_distance']

        # Plot
        subfigure = self._figure.add_subplot(self._gs[4, 0])
        subfigure.plot(x_spatial_axis, sinogram_max_var)
        subfigure.plot(zeros, np.zeros((len(zeros))), 'ro')
        subfigure.plot(x_spatial_axis[(x_spatial_axis >= zeros[0]) & (x_spatial_axis < zeros[-1])][max_indices],
                       sinogram_max_var[(x_spatial_axis >= zeros[0]) & (x_spatial_axis < zeros[-1])][max_indices], 'go')
        subfigure.plot(dx, np.zeros(len(dx)), 'ko')
        plt.title(
            r'Projected sinogram at $\theta$' +
            '= {:.1f} °'.format(
                self.metrics['direction']))
        plt.xlabel(r'$\rho$ (m)')
        plt.ylabel('Corr')
        plt.grid(True, linestyle='--', linewidth=0.5)

    def show_failed_sinogram(self) -> None:
        """ Show sinogram on which computation has failed
        """
        # Import sinogram
        sinogram_max_var = self.metrics['sinogram_max_var']

        # Retrieve axis of the sinogram value plot
        spatial_res = self.metrics['spatial_resolution']
        x_spatial_axis = np.arange(-(len(sinogram_max_var) // 2),
                                   len(sinogram_max_var) // 2 + 1) * spatial_res

        # Plot
        subfigure = self._figure.add_subplot(self._gs[4, 0])
        subfigure.plot(x_spatial_axis, sinogram_max_var)
        plt.xlabel(r'$\rho$ (m)')
        plt.ylabel('Corr')
        plt.grid('on')

    def show_depth_esti_values(self) -> None:
        """ Show physical values for a debug point
        """
        bathymetry_estimation = self.metrics['bathymetry_estimation']

        celerities_txt = str(['{:.2f}'.format(elem)
                             for elem in bathymetry_estimation.get_attribute('celerity')])
        periods_txt = str(['{:.2f}'.format(elem)
                          for elem in bathymetry_estimation.get_attribute('period')])
        depth_txt = str(['{:.2f}'.format(elem)
                        for elem in bathymetry_estimation.get_attribute('depth')])

        subfigure = self._figure.add_subplot(self._gs[4, 1])
        subfigure.axis('off')

        if self.bathymetry_estimations:
            subfigure.annotate('For time lag = {:.3f} s\n'.format(self.metrics['propagation_duration']) +
                               'Estimated wave:\n' +
                               r'$\theta$ = {:.1f}° from East'.format(self.metrics['direction']) + '\n' +
                               'L = {:.2f} m \n'.format(self.bathymetry_estimations[0].wavelength) +
                               'T = {:s} s \n'.format(periods_txt) +
                               'c = {:s} m/s \n'.format(celerities_txt) +
                               'H = {:s} m'.format(depth_txt),
                               xy=(0, 0),
                               # xytext=(-0.25,0.25),
                               xytext=(0, 0.05),
                               color='r',
                               fontweight='bold',
                               bbox=dict(boxstyle='round', facecolor='white', edgecolor='red')
                               )
        else:
            subfigure.annotate(
                'For time lag = {:.3f} s\n'.format(
                    self.metrics['propagation_duration']) + 'ESTIMATION FAILED !', xy=(
                    0, 0), xytext=(
                    0, 0.35), color='r', fontweight='bold', bbox=dict(
                    boxstyle='round', facecolor='white', edgecolor='red'))

    def show_values(self) -> None:
        """ Construct debug log and print it for a debug point
        """

        # Retreive values to add in the log
        bathymetry_estimation = self.metrics['bathymetry_estimation']
        time_lag = self.metrics['propagation_duration']

        celerities = bathymetry_estimation.get_attribute('celerity')
        celerities_txt = str(['{:.2f}'.format(elem) for elem in celerities])

        distances = bathymetry_estimation.get_attribute('delta_position')
        distances_txt = str(['{:.2f}'.format(elem) for elem in distances])

        linerities = bathymetry_estimation.get_attribute('linearity')
        linerities_txt = str(['{:.2f}'.format(elem) for elem in linerities])

        spatial_res = self.ortho_sequence[0].resolution
        wind_shape = self.ortho_sequence[0].pixels.shape
        wind_size = tuple(['{:.3f} m'.format(val * spatial_res) for val in wind_shape])

        # Construct the log

        txt = ['Debug information: \n' +
               '  S2Shore config: \n' +
               '    Window size (m): {:.3f} m \n'.format(self.global_estimator.window_size_x) +
               '    Lag number: {:d} frames\n'.format(self.local_estimator_params['TEMPORAL_LAG']) +
               '    Percentage points: {:d} % \n \n'.format(self.local_estimator_params['PERCENTAGE_POINTS']) +
               '  Image info:\n' +
               '    center point: ' + str(bathymetry_estimation.location) + '\n' +
               '    Window size (m): ' + str(wind_size) + '\n' +
               '    Spatial res: {:.3f} m \n'.format(spatial_res) +
               '    Window shape (px): ' + str(wind_shape) + '\n \n' +
               '  Temporal correlation info:\n' +
               '    time lag: {:.2f} s \n'.format(time_lag) +
               '    max var angle: {:.1f}° (direction from East)\n'.format(self.metrics['direction']) +
               '    estimated wavelength: {:.2f} m\n'.format(bathymetry_estimation[0].wavelength) +
               '    dx (m): {:s} \n'.format(distances_txt) +
               '    c (m/s): {:s} \n'.format(celerities_txt) +
               '    gamma: {:s} \n'.format(linerities_txt) +
               '    status: {:d}'.format(self.metrics['status']) +
               ' (0: SUCCESS, 1: FAIL, 2: ON_GROUND, 3: NO_DATA, 4: NO_DELTA_TIME, 5: OUTSIDE_ROI, 6: BEYOND_OFFSHORE_LIMIT) \n' +
               'End of debug \n'
               ]

        self._debug_log = txt[0]

        # Print the log
        print('')
        print(txt[0])

    def print_correlation_matrix_error(self) -> None:
        """ Display a message for correlation matrix error in debug image
        """
        subfigure = self._figure.add_subplot(self._gs[1, :2])
        subfigure.axis('off')
        subfigure.annotate('Correlation can not be computed',
                           (0, 0), color='g')

    def dump_figure(self) -> None:
        """ Save figure for a debug point
        """
        if self.global_estimator.debug_path:
            self._figure.savefig(os.path.join(self.global_estimator.debug_path,
                                              f'Infos_point_{self.location.x}_{self.location.y}.png'
                                              ), dpi=600
                                 )
        plt.close()

    def dump_debug_log(self) -> None:
        """ Save log for a debug point
        """
        if self.global_estimator.debug_path:
            debug_log_path = os.path.join(self.global_estimator.debug_path,
                                          f'Infos_point_{self.location.x}_{self.location.y}.txt'
                                          )

            with open(debug_log_path, 'w') as file:
                file.write(self._debug_log)

    def explore_results(self) -> None:
        """ Full routine for debugging point
        """
        self.show_first_frame()
        self.show_first_frame_selection()
        self.show_first_TS()
        self.show_correlation_matrix()
        self.show_correlation_matrix_filled_filtered()
        self.show_radon_matrix()
        self.show_sinogram()
        self.show_depth_esti_values()
        self.show_values()
        self.dump_figure()
        self.dump_debug_log()
