# -*- coding: utf-8 -*-
""" Module containing a modified radon transform compatible with skimage.transform.radon, but able
to manage tomographic angles in several ways.

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: Thu Apr 1 2021

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
from functools import lru_cache
from typing import List, Optional, Tuple, Union, cast  # @NoMove
from warnings import warn

import numpy as np
from skimage.transform import warp
from skimage.util.dtype import img_as_float


# TODO: use DirectionsQuantizer
# TODO: add quantized directions in the output
def symmetric_radon(image: np.ndarray,
                    theta: Optional[np.ndarray] = None,
                    circle: bool = True,
                    *,
                    preserve_range: bool = False) -> np.ndarray:
    """ Calculates the radon transform [1]_ [2]_ of an image given specified
    projection angles.

    Parameters
    ----------
    image : array_like
        Input image. The rotation axis will be located in the pixel with
        indices ``(image.shape[0] // 2, image.shape[1] // 2)``.
    theta : array_like, optional
        Projection angles (in degrees). If `None`, the value is set to
        np.arange(180).
    circle : boolean, optional
        Assume image is zero outside the inscribed circle, making the
        width of each projection (the first dimension of the sinogram)
        equal to ``min(image.shape)``.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html

    Returns
    -------
    radon_image : ndarray
        Radon transform (sinogram).  The tomography rotation axis will lie
        at the pixel index ``radon_image.shape[0] // 2`` along the 0th
        dimension of ``radon_image``.

    References
    ----------
    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
           Imaging", IEEE Press 1988.
    .. [2] B.R. Ramesh, N. Srinivasa, K. Rajgopal, "An Algorithm for Computing
           the Discrete Radon Transform With Some Applications", Proceedings of
           the Fourth IEEE Region 10 International Conference, TENCON '89, 1989

    Notes
    -----
    Based on code of Justin K. Romberg
    (https://www.clear.rice.edu/elec431/projects96/DSP/bpanalysis.html)

    """
    if image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    if theta is None:
        theta = np.arange(180)

    if preserve_range:
        # Convert image to double only if it is not single or double
        # precision float
        if image.dtype.char not in 'df':
            image = image.astype(float)
    else:
        image = img_as_float(image)

    if circle:
        shape_min = min(image.shape)
        radius = shape_min // 2
        img_shape = np.array(image.shape)
        coords = np.array(np.ogrid[:image.shape[0], :image.shape[1]],
                          dtype=object)
        dist = ((coords - img_shape // 2) ** 2).sum(0)
        outside_reconstruction_circle = dist > radius ** 2
        if np.any(image[outside_reconstruction_circle]):
            warn('Radon transform: image must be zero outside the '
                 'reconstruction circle')
        # Crop image to make it square
        slices = tuple(slice(int(np.ceil(excess / 2)),
                             int(np.ceil(excess / 2) + shape_min))
                       if excess > 0 else slice(None)
                       for excess in (img_shape - shape_min))
        padded_image = image[slices]
    else:
        diagonal = np.sqrt(2) * max(image.shape)
        pad = [int(np.ceil(diagonal - s)) for s in image.shape]
        new_center = [(s + p) // 2 for s, p in zip(image.shape, pad)]
        old_center = [s // 2 for s in image.shape]
        pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
        pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
        padded_image = np.pad(image, pad_width, mode='constant',
                              constant_values=0)

    # padded_image is always square
    if padded_image.shape[0] != padded_image.shape[1]:
        raise ValueError('padded_image must be a square')
    center = padded_image.shape[0] // 2
    radon_image = np.zeros((padded_image.shape[0], len(theta)),
                           dtype=image.dtype)

    # split angles in 2 sets: one for angles which must be processed, the other for angles
    # which can be produced from the computed angles by a flip.
    angles_to_compute, angles_to_flip = get_angles_sets(theta)

    for i, angle in angles_to_compute:
        rotation_matrix = get_rotation_matrix(center, angle)
        rotated = warp(padded_image, rotation_matrix, clip=False)
        radon_image[:, i] = rotated.sum(0)

    for i, j in angles_to_flip:
        radon_image[:, i] = np.flip(radon_image[:, j])
    return radon_image


@lru_cache(maxsize=1024)
def get_rotation_matrix(center: float, theta: float) -> np.ndarray:
    """ Computes the rotation matrix to be applied to compute the radon transform for one angle

    :param center: the position of the rotation center in the image along its smallest dimension
    :param theta: the tomographic angle
    :returns: A 3*3 array expressing the rotation to apply to the image.
    """
    angle = np.deg2rad(theta)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                     [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                     [0, 0, 1]])


def get_angles_sets(theta: np.ndarray) -> Tuple[List[Tuple[int, float]], List[Tuple[int, int]]]:
    """ Split the projection angles in two sets: one containing angles which must be computed,
    and the other one containing angles which can be produced by a flip of the computed ones.

    :param theta: the set of tomographic angles
    :returns: The list of angles which must be computed together with their index in the projection
              The list of angles indexes which can be produced by a flip of another angle specified
              by its index.
    """

    normalized_angles = _normalize_angle(theta)
    angles = _quantize(normalized_angles)

    angles_indices_1 = np.argwhere(angles >= 0.)[:, 0]
    angles_indices_2 = np.argwhere(angles < 0.)[:, 0]

    if angles_indices_1.size < angles_indices_2.size:
        angles_indices_1, angles_indices_2 = angles_indices_2, angles_indices_1

    return _process_angles_subsets(angles, angles_indices_1, angles_indices_2)


def _process_angles_subsets(angles: np.ndarray, largest: np.ndarray, smallest: np.ndarray, ) \
        -> Tuple[List[Tuple[int, float]], List[Tuple[int, int]]]:
    """ Utility to manage angles split in both cases of largest angles set being positive
    or negative. The largest set must be computed and the smallest one must be either computed
    or must be produced by a flip of computed angles.

    :param angles: the set of tomographic angles to split
    :param largest: largest set of indices of angles of the same sign which must be computed
    :param smallest: remaining angles indices (their sign is opposite to the sign of angles
                     pointed by the largest set)
    :returns: The list of angles which must be computed together with their index in the projection
              The list of angles indexes which can be produced by a flip of another angle specified
              by its index.
    """
    angles_to_flip = []
    angles_to_compute = [(index, angles[index]) for index in largest]
    for index_angle in smallest:
        angle = angles[index_angle]
        if angle >= 0.:
            opposite_angle = angle - 180.
        else:
            opposite_angle = angle + 180.
        quantized_opposite_angle = _quantize(opposite_angle)
        if quantized_opposite_angle in angles[largest]:
            index_opposite_angle = np.where(angles == quantized_opposite_angle)[0][0]
            angles_to_flip.append((index_angle, index_opposite_angle))
        else:
            angles_to_compute.append((index_angle, angle))
    return angles_to_compute, angles_to_flip


def _normalize_angle(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """ Normalize angle(s) expressed in degrees to the interval [-180°, 180°[

    :param angle: the real valued angle(s) expressed in degrees
    :returns: multiple(s) of quantization_step such that:
              quantized_value - quantization_step/2 < value <= quantized_value + quantization_step/2
    """
    # Limit angle between 0 and 360 degrees
    normalized_angle = angle % 360.
    # angle between -180 and +180 degrees
    if isinstance(normalized_angle, float):
        # Real case
        if normalized_angle >= 180.:
            normalized_angle -= 360.
    else:
        # np.ndarray case
        normalized_angle[normalized_angle >= 180] -= 360.
    return normalized_angle


def _quantize(value: Union[float, np.ndarray],
              quantization_step: float = cast(float, 0.1)) -> Union[float, np.ndarray]:
    """ Quantize real numbers such that 0. is the center of the quantization scale.

    :param value: the real value(s) to quantize
    :param quantization_step: the quantization interval length
    :returns: multiple(s) of quantization_step such that:
              quantized_value - quantization_step/2 < value <= quantized_value + quantization_step/2
    """
    if isinstance(value, float):
        half_delta = np.sign(value) * quantization_step / 2.
        index = int((value + half_delta) / quantization_step)
    else:
        index = np.rint(value / quantization_step)
    return index * quantization_step
