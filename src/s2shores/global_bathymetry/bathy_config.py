"""
Definition of the BathyConfig class

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 07/04/2025

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
from typing import Literal

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

from s2shores import __version__

class GlobalEstimatorConfig(BaseModel):

    WAVE_EST_METHOD: Literal["SPATIAL_DFT", "TEMPORAL_CORRELATION", "SPATIAL_CORRELATION"]
    SELECTED_FRAMES: list[str | int] | None = None

    OUTPUT_FORMAT: Literal["POINT", "GRID"]
    DXP: PositiveFloat
    DYP: PositiveFloat
    LAYERS_TYPE: Literal["NOMINAL", "EXPERT", "DEBUG"]
    NKEEP: PositiveInt
    OFFSHORE_LIMIT: PositiveFloat

    WINDOW: PositiveFloat
    SM_LENGTH: PositiveInt

    MIN_D: PositiveFloat
    MIN_T: PositiveFloat
    MAX_T: PositiveFloat
    MIN_WAVES_LINEARITY: PositiveFloat
    MAX_WAVES_LINEARITY: PositiveFloat

    DEPTH_EST_METHOD: Literal["LINEAR"]


class DebugPlotConfig(BaseModel):

    PLOT_MAX: float
    PLOT_MIN: float


class SpatialDFTConfig(BaseModel):
    
    PROMINENCE_MAX_PEAK: PositiveFloat
    PROMINENCE_MULTIPLE_PEAKS: PositiveFloat
    UNWRAP_PHASE_SHIFT: bool
    ANGLE_AROUND_PEAK_DIR: PositiveFloat
    STEP_T: PositiveFloat
    DEBUG: DebugPlotConfig


class TemporalCorrelationTuningConfig(BaseModel):

    DETREND_TIME_SERIES: Literal[0, 1]
    FILTER_TIME_SERIES: Literal[0, 1]
    LOWCUT_PERIOD: PositiveFloat
    HIGHCUT_PERIOD: PositiveFloat
    PEAK_DETECTION_HEIGHT_RATIO: PositiveFloat
    PEAK_DETECTION_DISTANCE_RATIO: PositiveFloat
    RATIO_SIZE_CORRELATION: PositiveFloat
    MEAN_FILTER_KERNEL_SIZE_SINOGRAM: PositiveInt
    SIGMA_CORRELATION_MASK: PositiveFloat
    MEDIAN_FILTER_KERNEL: PositiveInt
    

class TemporalCorrelationConfig(BaseModel):
    
    TEMPORAL_LAG: PositiveInt
    PERCENTAGE_POINTS: float = Field(ge=0, le=1)
    TUNING: TemporalCorrelationTuningConfig


class SpatialCorrelationConfig(BaseModel):
    
    CORRELATION_MODE: Literal["full", "valid", "same"]
    AUGMENTED_RADON_FACTOR: PositiveFloat
    PEAK_POSITION_MAX_FACTOR: PositiveFloat
    DEBUG: DebugPlotConfig


class BathyConfig(BaseModel):
    
    GLOBAL_ESTIMATOR: GlobalEstimatorConfig
    SPATIAL_DFT: SpatialDFTConfig
    TEMPORAL_CORRELATION: TemporalCorrelationConfig
    SPATIAL_CORRELATION: SpatialCorrelationConfig

    CHAINS_VERSIONS: str = f"s2shores : {__version__}"
