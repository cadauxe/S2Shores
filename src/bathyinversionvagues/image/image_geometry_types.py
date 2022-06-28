# -*- coding: utf-8 -*-
""" Definition of several types useful for static typing checks

:author: GIROS Alain
:created: 18/05/2021
"""
from typing import Tuple, Sequence

MarginsType = Tuple[float, float, float, float]

PointType = Tuple[float, float]

ImageWindowType = Tuple[int, int, int, int]

GdalGeoTransformType = Sequence[float]
