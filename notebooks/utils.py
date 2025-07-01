from pathlib import Path
from typing import Any, Callable
import yaml
from shapely.geometry import Point
from xarray import Dataset
import numpy as np

from s2shores.bathy_debug.waves_image_display import (
    create_pseudorgb,
    build_display_waves_image,
    build_display_pseudorgb,
)
from s2shores.bathylauncher.products.geotiff_product import GeoTiffProduct
from s2shores.bathylauncher.products.s2_image_product import S2ImageProduct
from s2shores.bathylauncher.bathy_launcher import BathyLauncher
from s2shores.data_providers.delta_time_provider import DeltaTimeProvider
from s2shores.global_bathymetry.bathy_config import BathyConfig
from s2shores.global_bathymetry.bathy_estimator import BathyEstimator
from s2shores.global_bathymetry.ortho_bathy_estimator import OrthoBathyEstimator
from s2shores.image.ortho_sequence import OrthoSequence
from s2shores.data_model.estimated_points_bathy import EstimatedPointsBathy
from s2shores.local_bathymetry.local_bathy_estimator import LocalBathyEstimator

import matplotlib.pyplot as plt

def initialize_sequential_run(
        product_path: Path,
        config: BathyConfig,
        delta_time_provider: DeltaTimeProvider = None,
) -> tuple[BathyEstimator, OrthoBathyEstimator]:
    bathy_launcher = BathyLauncher(cluster=None, sequential_run=True)
    bathy_estimator = initialize_bathy_estimator(
        bathy_launcher=bathy_launcher,
        product_path=product_path,
        output_path=...,
        config=config,
        delta_time_provider=delta_time_provider,
    )
    ortho_bathy_estimator = initialize_ortho_bathy_estimator(bathy_estimator)
    plot_whole_image(ortho_bathy_estimator)

    return bathy_estimator, ortho_bathy_estimator


def plot_whole_image(ortho_bathy_estimator: OrthoBathyEstimator, point: Point = None):
    img = ortho_bathy_estimator.sampled_ortho.read_frame_image(
        ortho_bathy_estimator.parent_estimator.selected_frames[0]).pixels
    print("Image shape in pixels : ",img.shape)
    ax_img = plt.imshow(img)
    origin = (
        ortho_bathy_estimator
        .sampled_ortho
        .ortho_stack
        ._geo_transform
        .image_coordinates(Point(0,0))
    )

    spatial_layout = ortho_bathy_estimator.sampled_ortho.ortho_stack
    spatial_xs = (str(spatial_layout._upper_left_corner.x),
                  str(spatial_layout._lower_right_corner.x))
    spatial_ys = (str(spatial_layout._lower_right_corner.y),
                  str(spatial_layout._upper_left_corner.y))


    if point is not None:
        image_point = (
            ortho_bathy_estimator
            .sampled_ortho
            .ortho_stack
            ._geo_transform
            .image_coordinates(point)
        )
        # plt.plot(image_point.y, img.shape[1] - image_point.x, "ro")
        # plt.plot(0, 0, "ro")
        # plt.plot(origin.y, origin.x, "bo")

        # print("POINT (0 0) ->", origin)
        # print(f"{point} -> {image_point}")

    ax_img.axes.set_xticks((0, img.shape[1]))
    ax_img.axes.set_xticklabels(spatial_xs)

    ax_img.axes.set_yticks((0, img.shape[0]))
    ax_img.axes.set_yticklabels(spatial_ys)


    # print(f"{img.shape} -> ({spatial_xs[1]}, {spatial_ys[1]})")


def initialize_bathy_estimator(
        bathy_launcher: BathyLauncher,
        product_path: Path,
        output_path: Path,
        config: BathyConfig,
        delta_time_provider: DeltaTimeProvider,
) -> BathyEstimator:
    match product_path.suffix:
        case ".tif":
            product_cls = GeoTiffProduct
        case ".SAFE":
            product_cls = S2ImageProduct
        case _:
            raise ValueError(
                "Product file type not recognized. Please use a geotiff or S2 file."
            )

    bathy_estimator = bathy_launcher.add_product(
        product_path.stem,
        product_path,
        product_cls,
        output_path,
        config.model_dump(),
        nb_subtiles=9,
    )

    bathy_estimator.set_delta_time_provider(delta_time_provider)
    bathy_estimator.create_subtiles()
    return bathy_estimator

def read_config(config_path: Path) -> BathyConfig:
    with config_path.open() as file:
        wave_params = yaml.load(file, Loader=yaml.FullLoader)

    return BathyConfig.model_validate(wave_params)


def initialize_ortho_bathy_estimator(
        bathy_estimator: BathyEstimator,
) -> OrthoBathyEstimator:
    subtile = bathy_estimator.subtiles[0]
    return OrthoBathyEstimator(bathy_estimator, subtile)


def build_ortho_sequence(
        ortho_bathy_estimator: OrthoBathyEstimator,
        estimation_point: Point,
) -> OrthoSequence:
    ortho_bathy_estimator.parent_estimator.set_debug_samples([estimation_point])
    ortho_bathy_estimator.parent_estimator._debug_sample = True
    window = ortho_bathy_estimator.sampled_ortho.window_extent(estimation_point)

    ortho_sequence = OrthoSequence(ortho_bathy_estimator.parent_estimator.delta_time_provider)
    for frame_id in ortho_bathy_estimator.parent_estimator.selected_frames:
        ortho_sequence.append_image(
            ortho_bathy_estimator.sampled_ortho.read_frame_image(frame_id),
            frame_id,
        )

    return ortho_sequence.extract_window(window)


def build_dataset(
        bathy_estimator: BathyEstimator,
        ortho_bathy_estimator: OrthoBathyEstimator,
        local_bathy_estimator: LocalBathyEstimator,
) -> Dataset:
    estimated_bathy = EstimatedPointsBathy(
        1,
        ortho_bathy_estimator.sampled_ortho.ortho_stack.acquisition_time,
    )
    estimated_bathy.store_estimations(0, local_bathy_estimator.bathymetry_estimations)

    dataset = estimated_bathy.build_dataset(
        bathy_estimator.layers_type,
        bathy_estimator.nb_max_wave_fields,
    )

    dataset = dataset.assign(spatial_ref=bathy_estimator._ortho_stack.build_spatial_ref())

    # necessary to have a correct georeferencing
    if 'x' in dataset.coords:  # only if output_format is GRID
        dataset.x.attrs['standard_name'] = 'projection_x_coordinate'
        dataset.y.attrs['standard_name'] = 'projection_y_coordinate'

    infos = bathy_estimator.build_infos()
    infos.update(bathy_estimator._ortho_stack.build_infos())
    for key, value in infos.items():
        dataset.attrs[key] = value

    return dataset


def plot_waves_row(
        fig,
        axs,
        row_number: int,
        pixels1: np.ndarray,
        resolution1: float,
        pixels2: np.ndarray,
        resolution2: float,
        nrows: int,
        ncols: int,
        title_suffix: str = "",
        directions: list[tuple[float, int]] = None,
        ):
    build_display_waves_image(fig,
                              axs[row_number, 0],
                              f'Image1{title_suffix}',
                              pixels1,
                              resolution=resolution1,
                              subplot_pos=[nrows, ncols, 1 + 3 * row_number],
                              cmap='gray',
                              directions=directions)
    build_display_pseudorgb(fig,
                            axs[row_number, 1],
                            f'Pseudo RGB{title_suffix}',
                            create_pseudorgb(pixels1, pixels2),
                            resolution=resolution1,
                            subplot_pos=[nrows, ncols, 2 + 3 * row_number],
                            coordinates=False)
    build_display_waves_image(fig,
                              axs[row_number, 2],
                              f'Image2{title_suffix}',
                              pixels2,
                              resolution=resolution2,
                              subplot_pos=[nrows, ncols, 3 + 3 * row_number],
                              directions=directions,
                              cmap='gray',
                              coordinates=False)