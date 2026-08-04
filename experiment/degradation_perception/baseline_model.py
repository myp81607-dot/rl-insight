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

"""Fit, validate, persist, and restore immutable KDE baseline models.

This module is an adapter around the existing degradation algorithm.  It does
not implement KDE fitting, stable-segment discovery, threshold ratios,
interval rules, or association analysis.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .algorithm import DegradationPerception, build_standard_data
from .perception_config import (
    SUPPORTED_SOURCE_TYPES,
    DetectionResult,
    ThresholdModel,
    TimeSeries,
)
from .policy import (
    overlay_metric_policy,
    resolve_history_policy,
    resolve_metric_policy,
)
from .serialization import to_json_serializable
from .timeseries import validate_detection_input


SCHEMA_VERSION = 1

_BASELINE_BOUND_POLICY_KEYS = {
    "alpha",
    "upper_ratio",
    "lower_ratio",
    "minimum_standard_points",
    "normalization",
    "kde",
    "stable_segment",
}


def _metric_names(metrics: Sequence[str] | str) -> list[str]:
    raw = [metrics] if isinstance(metrics, str) else list(metrics)
    selected: list[str] = []
    for metric in raw:
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError("metrics must contain at least one non-empty name")
        if metric not in selected:
            selected.append(metric)
    if not selected:
        raise ValueError("metrics must contain at least one non-empty name")
    return selected


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _reject_nonfinite(value: Any, location: str = "baseline model") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{location}[{index}]")


def _parse_constant(value: str) -> None:
    raise ValueError(f"baseline model contains invalid JSON number {value}")


def _detection_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in policy.items()
        if key != "association"
    }


def _changed_policy_paths(
    saved: Mapping[str, Any],
    override: Mapping[str, Any],
    prefix: str = "",
) -> list[str]:
    changed: list[str] = []
    for key, value in override.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        saved_value = saved.get(key)
        if isinstance(value, Mapping) and isinstance(saved_value, Mapping):
            changed.extend(_changed_policy_paths(saved_value, value, path))
        elif saved_value != value:
            changed.append(path)
    return changed


def _validate_baseline_bound_overrides(
    metric: str,
    saved: Mapping[str, Any],
    override: Mapping[str, Any],
) -> None:
    baseline_override = {
        key: value
        for key, value in override.items()
        if key in _BASELINE_BOUND_POLICY_KEYS
    }
    changed = _changed_policy_paths(saved, baseline_override)
    if changed:
        raise ValueError(
            f"algorithm policy changes fitted baseline parameters for {metric!r}: "
            f"{', '.join(changed)}; run fit again before detection"
        )


@dataclass(frozen=True)
class MetricBaseline:
    """One original metric name, its resolved policy, and all normal modes."""

    metric: str
    policy: dict[str, Any]
    standard_samples: int
    models: tuple[ThresholdModel, ...]

    @property
    def detection_policy(self) -> dict[str, Any]:
        return _detection_policy(self.policy)

    def to_dict(self) -> dict[str, Any]:
        alpha = float(self.policy["alpha"])
        lower_ratio = float(self.policy["lower_ratio"])
        upper_ratio = float(self.policy["upper_ratio"])
        return {
            "alpha": alpha,
            "lowerProbability": alpha,
            "upperProbability": 1.0 - alpha,
            "lowerRatio": lower_ratio,
            "upperRatio": upper_ratio,
            "standardSamples": self.standard_samples,
            "metricPolicy": copy.deepcopy(self.policy),
            "detectionPolicy": self.detection_policy,
            "normalModels": [
                {
                    "modelId": model.mode_id,
                    "stableSegmentStepRange": {
                        "startStep": model.segment_start_time,
                        "endStep": model.segment_end_time,
                    },
                    "standardSegmentSamples": model.point_count,
                    "rawLowerThreshold": model.lower_kde_threshold,
                    "rawUpperThreshold": model.upper_kde_threshold,
                    "finalLowerThreshold": model.lower_threshold,
                    "finalUpperThreshold": model.upper_threshold,
                    "lowerRatio": lower_ratio,
                    "upperRatio": upper_ratio,
                    "bandwidth": model.bandwidth,
                    "diagnostics": copy.deepcopy(model.diagnostics),
                }
                for model in self.models
            ],
        }


@dataclass(frozen=True)
class BaselineModelFile:
    """Versioned baseline artifact shared by offline and online detection."""

    baseline_id: str
    created_at: str
    source_type: str
    metrics: dict[str, MetricBaseline]
    history_policy: dict[str, int]
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "baselineId": self.baseline_id,
            "createdAt": self.created_at,
            "sourceType": self.source_type,
            "historyPolicy": dict(self.history_policy),
            "metrics": {
                metric: baseline.to_dict()
                for metric, baseline in self.metrics.items()
            },
        }


def _validate_created_at(value: Any) -> str:
    text = _required_text(value, "createdAt")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("createdAt must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("createdAt must include a timezone")
    return text


def _load_metric_baseline(metric: str, raw: Any) -> MetricBaseline:
    if not isinstance(raw, Mapping):
        raise ValueError(f"metrics[{metric!r}] must be an object")

    alpha = _finite_number(raw.get("alpha"), f"metrics[{metric!r}].alpha")
    lower_probability = _finite_number(
        raw.get("lowerProbability"),
        f"metrics[{metric!r}].lowerProbability",
    )
    upper_probability = _finite_number(
        raw.get("upperProbability"),
        f"metrics[{metric!r}].upperProbability",
    )
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"metrics[{metric!r}].alpha must be between 0 and 0.5")
    if lower_probability != alpha or upper_probability != 1.0 - alpha:
        raise ValueError(
            f"metrics[{metric!r}] probabilities do not match alpha"
        )

    raw_models = raw.get("normalModels")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError(f"metrics[{metric!r}].normalModels must not be empty")

    models: list[ThresholdModel] = []
    ratios: list[tuple[float, float]] = []
    for index, item in enumerate(raw_models):
        location = f"metrics[{metric!r}].normalModels[{index}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{location} must be an object")
        mode_id = _integer(item.get("modelId"), f"{location}.modelId")
        raw_lower = _finite_number(
            item.get("rawLowerThreshold"), f"{location}.rawLowerThreshold"
        )
        raw_upper = _finite_number(
            item.get("rawUpperThreshold"), f"{location}.rawUpperThreshold"
        )
        final_lower = _finite_number(
            item.get("finalLowerThreshold"), f"{location}.finalLowerThreshold"
        )
        final_upper = _finite_number(
            item.get("finalUpperThreshold"), f"{location}.finalUpperThreshold"
        )
        if raw_lower >= raw_upper:
            raise ValueError(f"{location} raw lower threshold must be < upper")
        if final_lower >= final_upper:
            raise ValueError(f"{location} final lower threshold must be < upper")
        lower_ratio = _finite_number(
            item.get("lowerRatio", raw.get("lowerRatio", 1.15)),
            f"{location}.lowerRatio",
        )
        upper_ratio = _finite_number(
            item.get("upperRatio", raw.get("upperRatio", 1.15)),
            f"{location}.upperRatio",
        )
        if lower_ratio < 1.0 or upper_ratio < 1.0:
            raise ValueError(f"{location} ratios must be at least 1")
        bandwidth = _finite_number(item.get("bandwidth"), f"{location}.bandwidth")
        if bandwidth <= 0.0:
            raise ValueError(f"{location}.bandwidth must be positive")
        ratios.append((lower_ratio, upper_ratio))

        segment = item.get("stableSegmentStepRange", {})
        if not isinstance(segment, Mapping):
            raise ValueError(f"{location}.stableSegmentStepRange must be an object")
        segment_start = _finite_number(
            segment.get("startStep", 0.0), f"{location}.stableSegmentStepRange.startStep"
        )
        segment_end = _finite_number(
            segment.get("endStep", segment_start),
            f"{location}.stableSegmentStepRange.endStep",
        )
        if segment_start > segment_end:
            raise ValueError(f"{location} stable segment is reversed")
        point_count = _integer(
            item.get("standardSegmentSamples", 0),
            f"{location}.standardSegmentSamples",
        )
        diagnostics = item.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            raise ValueError(f"{location}.diagnostics must be an object")
        models.append(
            ThresholdModel(
                mode_id=mode_id,
                segment_start_time=segment_start,
                segment_end_time=segment_end,
                point_count=point_count,
                lower_kde_threshold=raw_lower,
                upper_kde_threshold=raw_upper,
                lower_threshold=final_lower,
                upper_threshold=final_upper,
                bandwidth=bandwidth,
                diagnostics=to_json_serializable(dict(diagnostics)),
            )
        )

    if len({model.mode_id for model in models}) != len(models):
        raise ValueError(f"metrics[{metric!r}] contains duplicate modelId values")
    if len(set(ratios)) != 1:
        raise ValueError(f"metrics[{metric!r}] normal models have inconsistent ratios")
    lower_ratio, upper_ratio = ratios[0]

    raw_policy = raw.get("metricPolicy")
    if raw_policy is None:
        policy = resolve_metric_policy(
            metric,
            {
                "alpha": alpha,
                "lower_ratio": lower_ratio,
                "upper_ratio": upper_ratio,
            },
        )
    elif isinstance(raw_policy, Mapping):
        policy = resolve_metric_policy(metric, raw_policy)
    else:
        raise ValueError(f"metrics[{metric!r}].metricPolicy must be an object")
    if (
        float(policy["alpha"]) != alpha
        or float(policy["lower_ratio"]) != lower_ratio
        or float(policy["upper_ratio"]) != upper_ratio
    ):
        raise ValueError(f"metrics[{metric!r}] policy conflicts with model fields")

    saved_detection_policy = raw.get("detectionPolicy")
    if saved_detection_policy is not None:
        if not isinstance(saved_detection_policy, Mapping):
            raise ValueError(f"metrics[{metric!r}].detectionPolicy must be an object")
        if dict(saved_detection_policy) != _detection_policy(policy):
            raise ValueError(
                f"metrics[{metric!r}].detectionPolicy conflicts with metricPolicy"
            )

    standard_samples = _integer(
        raw.get("standardSamples", sum(model.point_count for model in models)),
        f"metrics[{metric!r}].standardSamples",
    )
    return MetricBaseline(
        metric=metric,
        policy=policy,
        standard_samples=standard_samples,
        models=tuple(models),
    )


def baseline_model_from_dict(raw: Any) -> BaselineModelFile:
    """Validate and restore one JSON-native baseline model mapping."""

    if not isinstance(raw, Mapping):
        raise ValueError("baseline model root must be an object")
    _reject_nonfinite(raw)
    schema_version = raw.get("schemaVersion")
    if isinstance(schema_version, bool) or schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schemaVersion: {schema_version!r}; expected {SCHEMA_VERSION}"
        )
    baseline_id = _required_text(raw.get("baselineId"), "baselineId")
    created_at = _validate_created_at(raw.get("createdAt"))
    source_type = _required_text(raw.get("sourceType"), "sourceType")
    if source_type not in SUPPORTED_SOURCE_TYPES:
        raise ValueError(f"unsupported sourceType: {source_type!r}")
    history_raw = raw.get("historyPolicy")
    if history_raw is not None and not isinstance(history_raw, Mapping):
        raise ValueError("historyPolicy must be an object")
    history_policy = resolve_history_policy(history_raw)

    raw_metrics = raw.get("metrics")
    if not isinstance(raw_metrics, Mapping) or not raw_metrics:
        raise ValueError("baseline model metrics must be a non-empty object")
    metrics: dict[str, MetricBaseline] = {}
    for metric, value in raw_metrics.items():
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError("baseline model metric names must be non-empty strings")
        metrics[metric] = _load_metric_baseline(metric, value)
    return BaselineModelFile(
        baseline_id=baseline_id,
        created_at=created_at,
        source_type=source_type,
        metrics=metrics,
        history_policy=history_policy,
        schema_version=SCHEMA_VERSION,
    )


def fit_baseline_model(
    standard: Mapping[str, TimeSeries],
    metrics: Sequence[str] | str,
    *,
    baseline_id: str = "default",
    source_type: str = "training_log",
    metric_policies: Mapping[str, Mapping[str, Any]] | None = None,
    history_policy: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> BaselineModelFile:
    """Use the existing KDE/stable-segment implementation to fit a baseline."""

    selected = _metric_names(metrics)
    if not isinstance(standard, Mapping):
        raise TypeError("standard must be keyed by metric")
    canonical = validate_detection_input(
        {"standard": dict(standard), "inference": {}}
    )["standard"]
    if source_type not in SUPPORTED_SOURCE_TYPES:
        raise ValueError(f"unsupported source_type: {source_type!r}")
    baseline_id = _required_text(baseline_id, "baseline_id")
    resolved_history = resolve_history_policy(history_policy)
    overrides = dict(metric_policies or {})
    fitted: dict[str, MetricBaseline] = {}
    for metric in selected:
        policy = resolve_metric_policy(metric, overrides.get(metric))
        series = canonical.get(metric, TimeSeries())
        minimum = int(policy["minimum_standard_points"])
        if len(series) < minimum:
            raise ValueError(
                f"standard log has fewer than {minimum} points for {metric!r}"
            )
        models = build_standard_data(series.timestamps, series.values, policy)
        if not models:
            raise ValueError(
                f"standard data contains no trusted stable segment for {metric!r}"
            )
        fitted[metric] = MetricBaseline(
            metric=metric,
            policy=policy,
            standard_samples=len(series),
            models=tuple(copy.deepcopy(models)),
        )
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    model = BaselineModelFile(
        baseline_id=baseline_id,
        created_at=_validate_created_at(timestamp),
        source_type=source_type,
        metrics=fitted,
        history_policy=resolved_history,
    )
    return baseline_model_from_dict(model.to_dict())


def serialize_baseline_model(model: BaselineModelFile) -> str:
    """Return standards-compliant JSON with Python's full float precision."""

    if not isinstance(model, BaselineModelFile):
        raise TypeError("model must be a BaselineModelFile")
    validated = baseline_model_from_dict(model.to_dict())
    payload = to_json_serializable(validated.to_dict())
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)


def save_baseline_model(
    model: BaselineModelFile,
    output: str | Path,
    *,
    backup_existing: bool = True,
) -> Path:
    """Validate, write a same-directory temp file, then atomically replace."""

    serialized = serialize_baseline_model(model) + "\n"
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=destination.name + ".",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name

        if destination.exists() and backup_existing:
            history_dir = destination.parent / "history"
            history_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            backup = history_dir / f"{destination.stem}_{stamp}{destination.suffix}"
            backup_temp = history_dir / (backup.name + ".tmp")
            try:
                shutil.copyfile(destination, backup_temp)
                os.replace(backup_temp, backup)
            finally:
                if backup_temp.exists():
                    backup_temp.unlink()
        os.replace(temporary_name, destination)
        temporary_name = None
        return destination
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def load_baseline_model(
    path: str | Path,
    *,
    expected_metrics: Sequence[str] | str | None = None,
) -> BaselineModelFile:
    """Load a strict baseline artifact; never fall back to refitting."""

    source = Path(path)
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_parse_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"could not load baseline model {source}: {exc}") from exc
    model = baseline_model_from_dict(raw)
    if expected_metrics is not None:
        selected = _metric_names(expected_metrics)
        missing = [metric for metric in selected if metric not in model.metrics]
        if missing:
            raise ValueError(
                "requested metrics are absent from baseline model: "
                + ", ".join(repr(metric) for metric in missing)
            )
    return model


def baseline_model_sha256(model: BaselineModelFile) -> str:
    """Return a stable content fingerprint for monitor-state compatibility."""

    return hashlib.sha256(serialize_baseline_model(model).encode("utf-8")).hexdigest()


class BaselineDetectionRunner:
    """Run inference against restored normal models without fitting KDE."""

    def __init__(
        self,
        baseline: BaselineModelFile,
        *,
        metrics: Sequence[str] | str | None = None,
        task_id: str | None = None,
        association_targets: Sequence[str] | str | None = None,
        metric_policies: Mapping[str, Mapping[str, Any]] | None = None,
        history_policy: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(baseline, BaselineModelFile):
            raise TypeError("baseline must be a BaselineModelFile")
        validated = baseline_model_from_dict(baseline.to_dict())
        selected = (
            list(validated.metrics)
            if metrics is None
            else _metric_names(metrics)
        )
        missing = [metric for metric in selected if metric not in validated.metrics]
        if missing:
            raise ValueError(
                "requested metrics are absent from baseline model: "
                + ", ".join(repr(metric) for metric in missing)
            )
        self.baseline = validated
        self.metrics = selected
        self.standard: dict[str, TimeSeries] = {}
        overrides = dict(metric_policies or {})
        self.metric_policies: dict[str, dict[str, Any]] = {}
        for metric in selected:
            saved_policy = validated.metrics[metric].policy
            override = dict(overrides.get(metric, {}))
            _validate_baseline_bound_overrides(metric, saved_policy, override)
            self.metric_policies[metric] = overlay_metric_policy(
                metric,
                saved_policy,
                override,
            )
        self.detector = DegradationPerception(
            metrics=selected,
            task_id=task_id,
            source_type=validated.source_type,
            metric_policies=self.metric_policies,
            history_policy=(
                validated.history_policy
                if history_policy is None
                else history_policy
            ),
            association_targets=association_targets,
        )
        self.detector.standard_models = {
            metric: list(copy.deepcopy(validated.metrics[metric].models))
            for metric in selected
        }
        self.detector._standard_model_configs = {
            metric: _detection_policy(self.metric_policies[metric])
            for metric in selected
        }

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        metrics: Sequence[str] | str | None = None,
        task_id: str | None = None,
        association_targets: Sequence[str] | str | None = None,
        metric_policies: Mapping[str, Mapping[str, Any]] | None = None,
        history_policy: Mapping[str, Any] | None = None,
    ) -> "BaselineDetectionRunner":
        baseline = load_baseline_model(path, expected_metrics=metrics)
        return cls(
            baseline,
            metrics=metrics,
            task_id=task_id,
            association_targets=association_targets,
            metric_policies=metric_policies,
            history_policy=history_policy,
        )

    def minimum_inference_samples(self) -> dict[str, int]:
        return {
            metric: int(self.metric_policies[metric]["minimum_inference_points"])
            for metric in self.metrics
        }

    def snapshot_runtime_state(self) -> Any:
        """Snapshot mutable history so callers can roll back failed commits."""

        return copy.deepcopy(self.detector.history)

    def restore_runtime_state(self, snapshot: Any) -> None:
        self.detector.history = copy.deepcopy(snapshot)

    def run(self, inference: Mapping[str, TimeSeries]) -> DetectionResult:
        from .detection_runtime import (  # Avoid a module import cycle.
            build_detection_dataset,
            build_kde_debug_details,
        )

        dataset = build_detection_dataset({}, inference, self.metrics)
        result = self.detector.detect_dataset(dataset, metrics=self.metrics)
        debug = build_kde_debug_details(
            self.detector,
            result,
            dataset,
            self.metrics,
        )
        for metric in self.metrics:
            debug[metric]["standardSamples"] = self.baseline.metrics[
                metric
            ].standard_samples
        result["debugDetails"] = {"metrics": debug}  # type: ignore[typeddict-unknown-key]
        return to_json_serializable(result)


__all__ = [
    "BaselineDetectionRunner",
    "BaselineModelFile",
    "MetricBaseline",
    "SCHEMA_VERSION",
    "baseline_model_from_dict",
    "baseline_model_sha256",
    "fit_baseline_model",
    "load_baseline_model",
    "save_baseline_model",
    "serialize_baseline_model",
]
