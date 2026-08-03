# Copyright (c) 2026 verl-project authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RL-Insight degradation-perception algorithm module."""

from .algorithm import DegradationPerception, build_standard_data, get_standard_data
from .association_analysis import AssociationAnalyzer
from .perception_config import (
    DetectionInput,
    DetectionResult,
    MetricState,
    ThresholdModel,
    TimeSeries,
)
from .policy import (
    DEFAULT_HISTORY_POLICY,
    DEFAULT_METRIC_POLICY,
    resolve_history_policy,
    resolve_metric_policy,
)
from .stable_segment_detector import StableSegmentDetector
from .timeseries import preprocess_time_series, validate_detection_input
from .training_log import (
    TrainingLogParseResult,
    load_verl_training_log,
    parse_verl_training_log,
    parse_verl_training_log_line,
    parse_verl_training_log_with_metadata,
)

__all__ = [
    'TrainingLogParseResult',
    'load_verl_training_log',
    'parse_verl_training_log',
    'parse_verl_training_log_line',
    'parse_verl_training_log_with_metadata',
    "AssociationAnalyzer",
    "DEFAULT_HISTORY_POLICY",
    "DEFAULT_METRIC_POLICY",
    "DegradationPerception",
    "DetectionInput",
    "DetectionResult",
    "MetricState",
    "StableSegmentDetector",
    "ThresholdModel",
    "TimeSeries",
    "build_standard_data",
    "get_standard_data",
    "preprocess_time_series",
    "resolve_history_policy",
    "resolve_metric_policy",
    "validate_detection_input",
]
