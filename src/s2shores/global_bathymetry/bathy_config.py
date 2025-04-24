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

    OUTPUT_FORMAT: Literal["POINT", "GRID"] = "POINT"
    DXP: PositiveFloat = 5.
    DYP: PositiveFloat = 5.
    LAYERS_TYPE: Literal["NOMINAL", "EXPERT", "DEBUG"] = "DEBUG"
    NKEEP: PositiveInt = 3
    OFFSHORE_LIMIT: PositiveFloat = 100.

    WINDOW: PositiveFloat = 200.
    SM_LENGTH: PositiveInt = 10

    MIN_D: PositiveFloat = 0.5
    MIN_T: PositiveFloat = 5.
    MAX_T: PositiveFloat = 25.
    MIN_WAVES_LINEARITY: PositiveFloat = 0.1
    MAX_WAVES_LINEARITY: PositiveFloat = 1.

    DEPTH_EST_METHOD: Literal["LINEAR"] = "LINEAR"


class DebugPlotConfig(BaseModel):

    PLOT_MAX: float = 135.
    PLOT_MIN: float = -135.


class SpatialDFTConfig(BaseModel):
    
    PROMINENCE_MAX_PEAK: PositiveFloat = 0.3
    PROMINENCE_MULTIPLE_PEAKS: PositiveFloat = 0.1
    UNWRAP_PHASE_SHIFT: bool = False
    ANGLE_AROUND_PEAK_DIR: PositiveFloat = 10.
    STEP_T: PositiveFloat = 0.05
    DEBUG: DebugPlotConfig = DebugPlotConfig()


class TemporalCorrelationTuningConfig(BaseModel):

    DETREND_TIME_SERIES: Literal[0, 1] = 0
    FILTER_TIME_SERIES: Literal[0, 1] = 0
    LOWCUT_PERIOD: PositiveFloat = 25.
    HIGHCUT_PERIOD: PositiveFloat = 5.
    PEAK_DETECTION_HEIGHT_RATIO: PositiveFloat = 0.3
    PEAK_DETECTION_DISTANCE_RATIO: PositiveFloat= 0.5
    RATIO_SIZE_CORRELATION: PositiveFloat = 1.
    MEDIAN_FILTER_KERNEL_RATIO_SINOGRAM: PositiveFloat = 0.25
    MEAN_FILTER_KERNEL_SIZE_SINOGRAM: PositiveInt = 5
    SIGMA_CORRELATION_MASK: PositiveFloat = 2.
    MEDIAN_FILTER_KERNEL: PositiveInt = 5
    

class TemporalCorrelationConfig(BaseModel):
    
    TEMPORAL_LAG: PositiveInt = 1
    PERCENTAGE_POINTS: float = Field(20, ge=0, le=100)
    TUNING: TemporalCorrelationTuningConfig = TemporalCorrelationTuningConfig()


class SpatialCorrelationConfig(BaseModel):
    
    CORRELATION_MODE: Literal["full", "valid", "same"] = "full"
    AUGMENTED_RADON_FACTOR: PositiveFloat = 0.01
    PEAK_POSITION_MAX_FACTOR: PositiveFloat = 0.8
    DEBUG: DebugPlotConfig = DebugPlotConfig()


class BathyConfig(BaseModel):
    
    GLOBAL_ESTIMATOR: GlobalEstimatorConfig
    SPATIAL_DFT: SpatialDFTConfig = SpatialDFTConfig()
    TEMPORAL_CORRELATION: TemporalCorrelationConfig = TemporalCorrelationConfig()
    SPATIAL_CORRELATION: SpatialCorrelationConfig = SpatialCorrelationConfig()

    CHAINS_VERSIONS: str = f"s2shores : {__version__}"
