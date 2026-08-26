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

"""Typed configuration for the degradation experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from .association import AssociationParameters
from .baseline import BaselineParameters
from .detector import DetectorParameters
from .metrics import CANDIDATE_METRICS, TARGET_METRICS
from .prometheus import DEFAULT_GLOBAL_STEP_METRIC, DEFAULT_SERIES_SELECTOR
from .storage import ABNORMAL_DATA_NAME, STANDARD_DATA_NAME

DEFAULT_PROMETHEUS_URL = "http://127.0.0.1:9090"
DEFAULT_TARGET_METRICS = TARGET_METRICS
DEFAULT_CANDIDATE_METRICS = CANDIDATE_METRICS
DEFAULT_STANDARD_DATA_PATH = Path(STANDARD_DATA_NAME)
DEFAULT_ABNORMAL_DATA_PATH = Path(ABNORMAL_DATA_NAME)


def _positive_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _metric_names(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a sequence of metric names")
    if any(not isinstance(metric, str) for metric in value):
        raise TypeError(f"{name} must contain strings")
    metrics = tuple(metric.strip() for metric in value)
    if any(not metric for metric in metrics):
        raise ValueError(f"{name} must contain non-empty metric names")
    if len(set(metrics)) != len(metrics):
        raise ValueError(f"{name} must not contain duplicates")
    return metrics


@dataclass(frozen=True)
class RuntimeParameters:
    """Parameters used only by polling and orchestration."""

    baseline_step_count: int = 30
    poll_interval_seconds: float = 300.0
    query_step_seconds: float = 10.0
    pre_context_steps: int = 30
    top_k: int = 25
    target_metrics: tuple[str, ...] = DEFAULT_TARGET_METRICS
    candidate_metrics: tuple[str, ...] = DEFAULT_CANDIDATE_METRICS
    global_step_metric: str = DEFAULT_GLOBAL_STEP_METRIC
    series_selector: str = DEFAULT_SERIES_SELECTOR

    def __post_init__(self) -> None:
        for name in ("baseline_step_count", "pre_context_steps", "top_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.baseline_step_count <= 0:
            raise ValueError("baseline_step_count must be positive")
        if self.pre_context_steps < 0:
            raise ValueError("pre_context_steps must not be negative")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        _positive_number(self.poll_interval_seconds, "poll_interval_seconds")
        _positive_number(self.query_step_seconds, "query_step_seconds")

        targets = _metric_names(self.target_metrics, "target_metrics")
        candidates = _metric_names(self.candidate_metrics, "candidate_metrics")
        overlap = set(targets) & set(candidates)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"target_metrics and candidate_metrics overlap: {names}")
        object.__setattr__(self, "target_metrics", targets)
        object.__setattr__(self, "candidate_metrics", candidates)

        for name in ("global_step_metric", "series_selector"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())


@dataclass(frozen=True)
class DegradationConfig:
    """Complete dependency-free configuration for one runtime."""

    prometheus_url: str = DEFAULT_PROMETHEUS_URL
    standard_data_path: Path = DEFAULT_STANDARD_DATA_PATH
    abnormal_data_path: Path = DEFAULT_ABNORMAL_DATA_PATH
    request_timeout_seconds: float = 30.0
    verify_tls: bool = True
    runtime: RuntimeParameters = field(default_factory=RuntimeParameters)
    baseline: BaselineParameters = field(default_factory=BaselineParameters)
    detector: DetectorParameters = field(default_factory=DetectorParameters)
    association: AssociationParameters = field(default_factory=AssociationParameters)

    def __post_init__(self) -> None:
        if not isinstance(self.prometheus_url, str) or not self.prometheus_url.strip():
            raise ValueError("prometheus_url must be a non-empty string")
        object.__setattr__(self, "prometheus_url", self.prometheus_url.strip())
        object.__setattr__(self, "standard_data_path", Path(self.standard_data_path))
        object.__setattr__(self, "abnormal_data_path", Path(self.abnormal_data_path))
        if self.standard_data_path == self.abnormal_data_path:
            raise ValueError("standard and abnormal data paths must be different")
        _positive_number(self.request_timeout_seconds, "request_timeout_seconds")
        if not isinstance(self.verify_tls, bool):
            raise TypeError("verify_tls must be a boolean")
        expected = (
            (self.runtime, RuntimeParameters, "runtime"),
            (self.baseline, BaselineParameters, "baseline"),
            (self.detector, DetectorParameters, "detector"),
            (self.association, AssociationParameters, "association"),
        )
        for value, value_type, name in expected:
            if not isinstance(value, value_type):
                raise TypeError(f"{name} must be {value_type.__name__}")


__all__ = [
    "DEFAULT_ABNORMAL_DATA_PATH",
    "DEFAULT_CANDIDATE_METRICS",
    "DEFAULT_PROMETHEUS_URL",
    "DEFAULT_STANDARD_DATA_PATH",
    "DEFAULT_TARGET_METRICS",
    "DegradationConfig",
    "RuntimeParameters",
]
