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

"""Poll Prometheus and reuse the existing immutable-baseline detector."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml

from .algorithm_policy import (
    AlgorithmPolicyConfig,
    load_algorithm_policy_config,
)
from .baseline_model import (
    BaselineDetectionRunner,
    BaselineModelFile,
    baseline_model_sha256,
    fit_baseline_model,
    load_baseline_model,
    save_baseline_model,
)
from .perception_config import TimeSeries
from .prometheus_source import (
    PrometheusHttpClient,
    PrometheusMetricQuery,
    build_bootstrap_window,
    filter_after_global_step,
    latest_integer_step,
)
from .serialization import to_json_serializable


@dataclass(frozen=True)
class PrometheusConnectionConfig:
    url: str
    global_step_query: str
    query_step_seconds: float = 15.0
    lookback_seconds: float = 86400.0
    overlap_seconds: float = 30.0
    alignment_tolerance_seconds: float = 8.0
    timeout_seconds: float = 30.0
    verify_tls: bool = True
    bearer_token_env: str | None = None
    username_env: str | None = None
    password_env: str | None = None


@dataclass(frozen=True)
class BaselineBootstrapConfig:
    auto_fit: bool = True
    start_step: int = 5
    end_step: int = 25
    minimum_samples_per_metric: int = 10
    minimum_step_coverage_ratio: float = 0.5
    baseline_id: str | None = None


@dataclass(frozen=True)
class TargetSelectionConfig:
    patterns: tuple[str, ...] = ("time", "timing")
    case_sensitive: bool = False
    unmatched_role: str = "candidate"
    overrides: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PrometheusMonitorConfig:
    baseline_model: Path
    policy_config: Path
    prometheus: PrometheusConnectionConfig
    queries: tuple[PrometheusMetricQuery, ...]
    target_metrics: tuple[str, ...]
    candidate_metrics: tuple[str, ...]
    baseline: BaselineBootstrapConfig
    target_selection: TargetSelectionConfig
    poll_interval_seconds: float
    inference_window_points: int
    state_file: Path
    output_dir: Path
    task_id: str

    @property
    def metrics(self) -> tuple[str, ...]:
        return tuple(query.name for query in self.queries)


@dataclass
class PrometheusMonitorState:
    last_timestamp: float | None = None
    last_global_step: int | None = None
    inference_buffer: dict[str, TimeSeries] = field(default_factory=dict)
    round_id: int = 0
    active_event_ids: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PrometheusPollOutcome:
    status: str
    result: dict[str, Any] | None = None
    output_path: Path | None = None
    event_path: Path | None = None
    error: dict[str, str] | None = None


class DetectionRunner(Protocol):
    def minimum_inference_samples(self) -> dict[str, int]: ...

    def snapshot_runtime_state(self) -> Any: ...

    def restore_runtime_state(self, snapshot: Any) -> None: ...

    def run(self, inference: Mapping[str, TimeSeries]) -> dict[str, Any]: ...


_ROLES = {"auto", "target", "candidate", "ignore"}
_ABNORMAL_TYPES = {"UP", "DOWN", "BOTH"}


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive number")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return dict(value)


def _reject_unknown(
    raw: Mapping[str, Any],
    allowed: set[str],
    name: str,
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{name} contains unsupported fields: {', '.join(unknown)}")


def _optional_env_name(value: Any, name: str) -> str | None:
    return None if value is None else _required_text(value, name)


def _local_path(base: Path, value: Any, name: str) -> Path:
    raw = Path(_required_text(value, name)).expanduser()
    return raw.resolve() if raw.is_absolute() else (base / raw).resolve()


def _resolve_role(
    metric: str,
    configured_role: str,
    selection: TargetSelectionConfig,
) -> str:
    if configured_role != "auto":
        return configured_role
    if metric in selection.overrides:
        return selection.overrides[metric]
    candidate = metric if selection.case_sensitive else metric.lower()
    patterns = (
        selection.patterns
        if selection.case_sensitive
        else tuple(pattern.lower() for pattern in selection.patterns)
    )
    if any(pattern in candidate for pattern in patterns):
        return "target"
    return selection.unmatched_role


def load_prometheus_monitor_config(path: str | Path) -> PrometheusMonitorConfig:
    """Load the Prometheus runtime YAML without changing algorithm policy."""

    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"could not load Prometheus monitor config: {exc}") from exc
    root = _mapping(raw, "Prometheus monitor config root")
    _reject_unknown(
        root,
        {
            "baseline_model",
            "policy_config",
            "prometheus",
            "metrics",
            "target_selection",
            "baseline",
            "poll_interval_seconds",
            "inference_window_points",
            "state_file",
            "output_dir",
            "task_id",
        },
        "Prometheus monitor config",
    )
    base = source.parent

    prom_raw = _mapping(root.get("prometheus"), "prometheus")
    _reject_unknown(
        prom_raw,
        {
            "url",
            "global_step_query",
            "query_step_seconds",
            "lookback_seconds",
            "overlap_seconds",
            "alignment_tolerance_seconds",
            "timeout_seconds",
            "verify_tls",
            "bearer_token_env",
            "username_env",
            "password_env",
        },
        "prometheus",
    )
    query_step = _positive_float(
        prom_raw.get("query_step_seconds", 15),
        "prometheus.query_step_seconds",
    )
    prometheus = PrometheusConnectionConfig(
        url=_required_text(prom_raw.get("url"), "prometheus.url"),
        global_step_query=_required_text(
            prom_raw.get("global_step_query"),
            "prometheus.global_step_query",
        ),
        query_step_seconds=query_step,
        lookback_seconds=_positive_float(
            prom_raw.get("lookback_seconds", 86400),
            "prometheus.lookback_seconds",
        ),
        overlap_seconds=_positive_float(
            prom_raw.get("overlap_seconds", query_step * 2),
            "prometheus.overlap_seconds",
        ),
        alignment_tolerance_seconds=_positive_float(
            prom_raw.get("alignment_tolerance_seconds", query_step / 2 + 0.5),
            "prometheus.alignment_tolerance_seconds",
        ),
        timeout_seconds=_positive_float(
            prom_raw.get("timeout_seconds", 30),
            "prometheus.timeout_seconds",
        ),
        verify_tls=_boolean(
            prom_raw.get("verify_tls", True),
            "prometheus.verify_tls",
        ),
        bearer_token_env=_optional_env_name(
            prom_raw.get("bearer_token_env"),
            "prometheus.bearer_token_env",
        ),
        username_env=_optional_env_name(
            prom_raw.get("username_env"),
            "prometheus.username_env",
        ),
        password_env=_optional_env_name(
            prom_raw.get("password_env"),
            "prometheus.password_env",
        ),
    )
    if prometheus.overlap_seconds < prometheus.query_step_seconds:
        raise ValueError(
            "prometheus.overlap_seconds must be at least query_step_seconds"
        )

    selection_raw = _mapping(
        root.get("target_selection", {}),
        "target_selection",
    )
    _reject_unknown(
        selection_raw,
        {"patterns", "case_sensitive", "unmatched_role", "overrides"},
        "target_selection",
    )
    patterns_raw = selection_raw.get("patterns", ["time", "timing"])
    if not isinstance(patterns_raw, Sequence) or isinstance(patterns_raw, (str, bytes)):
        raise TypeError("target_selection.patterns must be an array")
    patterns = tuple(
        _required_text(item, "target_selection.patterns[]") for item in patterns_raw
    )
    if not patterns:
        raise ValueError("target_selection.patterns must not be empty")
    unmatched_role = _required_text(
        selection_raw.get("unmatched_role", "candidate"),
        "target_selection.unmatched_role",
    ).lower()
    if unmatched_role not in {"candidate", "ignore"}:
        raise ValueError("target_selection.unmatched_role must be candidate or ignore")
    overrides_raw = _mapping(
        selection_raw.get("overrides", {}),
        "target_selection.overrides",
    )
    overrides: dict[str, str] = {}
    for metric, role_value in overrides_raw.items():
        metric_name = _required_text(metric, "target_selection override metric")
        role = _required_text(
            role_value,
            f"target_selection.overrides.{metric_name}",
        ).lower()
        if role not in {"target", "candidate", "ignore"}:
            raise ValueError(
                f"target_selection override for {metric_name!r} must be "
                "target, candidate, or ignore"
            )
        overrides[metric_name] = role
    selection = TargetSelectionConfig(
        patterns=patterns,
        case_sensitive=_boolean(
            selection_raw.get("case_sensitive", False),
            "target_selection.case_sensitive",
        ),
        unmatched_role=unmatched_role,
        overrides=overrides,
    )

    metrics_raw = _mapping(root.get("metrics"), "metrics")
    if not metrics_raw:
        raise ValueError("metrics must contain at least one query")
    queries: list[PrometheusMetricQuery] = []
    targets: list[str] = []
    candidates: list[str] = []
    for raw_name, raw_spec in metrics_raw.items():
        name = _required_text(raw_name, "metric name")
        if isinstance(raw_spec, str):
            spec = {"promql": raw_spec}
        else:
            spec = _mapping(raw_spec, f"metrics.{name}")
        _reject_unknown(
            spec,
            {"promql", "role", "abnormal_type"},
            f"metrics.{name}",
        )
        configured_role = _required_text(
            spec.get("role", "auto"),
            f"metrics.{name}.role",
        ).lower()
        if configured_role not in _ROLES:
            raise ValueError(
                f"metrics.{name}.role must be auto, target, candidate, or ignore"
            )
        role = _resolve_role(name, configured_role, selection)
        if role == "ignore":
            continue
        abnormal_type_raw = spec.get("abnormal_type")
        abnormal_type = None
        if abnormal_type_raw is not None:
            abnormal_type = _required_text(
                abnormal_type_raw,
                f"metrics.{name}.abnormal_type",
            ).upper()
            if abnormal_type not in _ABNORMAL_TYPES:
                raise ValueError(
                    f"metrics.{name}.abnormal_type must be UP, DOWN, or BOTH"
                )
        queries.append(
            PrometheusMetricQuery(
                name=name,
                promql=_required_text(
                    spec.get("promql"),
                    f"metrics.{name}.promql",
                ),
                role=role,
                abnormal_type=abnormal_type,
            )
        )
        (targets if role == "target" else candidates).append(name)
    if not queries:
        raise ValueError("all configured metrics are ignored")
    if not targets:
        raise ValueError(
            "no target metric matched target_selection; configure a time/timing "
            "metric or an explicit target override"
        )

    baseline_raw = _mapping(root.get("baseline", {}), "baseline")
    _reject_unknown(
        baseline_raw,
        {
            "auto_fit",
            "start_step",
            "end_step",
            "minimum_samples_per_metric",
            "minimum_step_coverage_ratio",
            "baseline_id",
        },
        "baseline",
    )
    start_step = baseline_raw.get("start_step", 5)
    end_step = baseline_raw.get("end_step", 25)
    if isinstance(start_step, bool) or not isinstance(start_step, int):
        raise TypeError("baseline.start_step must be an integer")
    if isinstance(end_step, bool) or not isinstance(end_step, int):
        raise TypeError("baseline.end_step must be an integer")
    if start_step > end_step:
        raise ValueError("baseline.start_step must not exceed baseline.end_step")
    coverage = baseline_raw.get("minimum_step_coverage_ratio", 0.5)
    if isinstance(coverage, bool):
        raise TypeError("baseline.minimum_step_coverage_ratio must be between 0 and 1")
    try:
        coverage_number = float(coverage)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "baseline.minimum_step_coverage_ratio must be between 0 and 1"
        ) from exc
    if not math.isfinite(coverage_number) or not 0 < coverage_number <= 1:
        raise ValueError("baseline.minimum_step_coverage_ratio must be between 0 and 1")
    baseline_id_raw = baseline_raw.get("baseline_id")
    baseline = BaselineBootstrapConfig(
        auto_fit=_boolean(
            baseline_raw.get("auto_fit", True),
            "baseline.auto_fit",
        ),
        start_step=start_step,
        end_step=end_step,
        minimum_samples_per_metric=_positive_int(
            baseline_raw.get("minimum_samples_per_metric", 10),
            "baseline.minimum_samples_per_metric",
        ),
        minimum_step_coverage_ratio=coverage_number,
        baseline_id=(
            None
            if baseline_id_raw is None
            else _required_text(baseline_id_raw, "baseline.baseline_id")
        ),
    )
    return PrometheusMonitorConfig(
        baseline_model=_local_path(
            base,
            root.get("baseline_model"),
            "baseline_model",
        ),
        policy_config=_local_path(
            base,
            root.get("policy_config", "./algorithm_config.yaml"),
            "policy_config",
        ),
        prometheus=prometheus,
        queries=tuple(queries),
        target_metrics=tuple(targets),
        candidate_metrics=tuple(candidates),
        baseline=baseline,
        target_selection=selection,
        poll_interval_seconds=_positive_float(
            root.get("poll_interval_seconds", 180),
            "poll_interval_seconds",
        ),
        inference_window_points=_positive_int(
            root.get("inference_window_points", 100),
            "inference_window_points",
        ),
        state_file=_local_path(
            base,
            root.get("state_file", "./prometheus_output/monitor_state.json"),
            "state_file",
        ),
        output_dir=_local_path(
            base,
            root.get("output_dir", "./prometheus_output"),
            "output_dir",
        ),
        task_id=_required_text(
            root.get("task_id", "degradation-prometheus"), "task_id"
        ),
    )


def _metric_policies(
    policy: AlgorithmPolicyConfig,
    queries: Sequence[PrometheusMetricQuery],
) -> dict[str, dict[str, Any]]:
    policies = policy.for_metrics([query.name for query in queries])
    for query in queries:
        if query.abnormal_type is not None:
            policies[query.name]["abnormal_type"] = query.abnormal_type
    return policies


def _source_fingerprint(config: PrometheusMonitorConfig) -> str:
    payload = {
        "url": config.prometheus.url,
        "globalStepQuery": config.prometheus.global_step_query,
        "queryStepSeconds": config.prometheus.query_step_seconds,
        "alignmentToleranceSeconds": (config.prometheus.alignment_tolerance_seconds),
        "queries": [
            {
                "name": query.name,
                "promql": query.promql,
                "role": query.role,
                "abnormalType": query.abnormal_type,
            }
            for query in config.queries
        ],
        "targets": list(config.target_metrics),
        "baselineSteps": [config.baseline.start_step, config.baseline.end_step],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _copy_series(series: TimeSeries) -> TimeSeries:
    return TimeSeries(list(series.timestamps), list(series.values))


def _copy_buffer(buffer: Mapping[str, TimeSeries]) -> dict[str, TimeSeries]:
    return {metric: _copy_series(series) for metric, series in buffer.items()}


def _merge_series(left: TimeSeries, right: TimeSeries) -> TimeSeries:
    points = {
        float(timestamp): float(value)
        for timestamp, value in zip(left.timestamps, left.values)
    }
    points.update(
        {
            float(timestamp): float(value)
            for timestamp, value in zip(right.timestamps, right.values)
        }
    )
    ordered = sorted(points.items())
    return TimeSeries(
        timestamps=[timestamp for timestamp, _ in ordered],
        values=[value for _, value in ordered],
    )


def _trim_buffer(
    buffer: Mapping[str, TimeSeries],
    maximum_points: int,
) -> dict[str, TimeSeries]:
    result: dict[str, TimeSeries] = {}
    for metric, series in buffer.items():
        start = max(0, len(series) - maximum_points)
        result[metric] = TimeSeries(
            timestamps=list(series.timestamps[start:]),
            values=list(series.values[start:]),
        )
    return result


def _safe_task_name(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("._") or "task"


def _write_json_atomically(destination: Path, payload: Mapping[str, Any]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            to_json_serializable(dict(payload)),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n"
    )
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
        os.replace(temporary_name, destination)
        temporary_name = None
        return destination
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _state_payload(
    state: PrometheusMonitorState,
    *,
    metrics: Sequence[str],
    source_fingerprint: str,
    baseline_sha256: str,
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "sourceFingerprint": source_fingerprint,
        "baselineSha256": baseline_sha256,
        "lastTimestamp": state.last_timestamp,
        "lastGlobalStep": state.last_global_step,
        "roundId": state.round_id,
        "activeEventIds": dict(state.active_event_ids),
        "inferenceBuffer": {
            metric: {
                "timestamps": list(
                    state.inference_buffer.get(metric, TimeSeries()).timestamps
                ),
                "values": list(state.inference_buffer.get(metric, TimeSeries()).values),
            }
            for metric in metrics
        },
    }


def _load_state(
    path: Path,
    *,
    metrics: Sequence[str],
    source_fingerprint: str,
    baseline_sha256_value: str,
    window_points: int,
) -> PrometheusMonitorState:
    empty = {metric: TimeSeries() for metric in metrics}
    if not path.exists():
        return PrometheusMonitorState(inference_buffer=empty)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load Prometheus monitor state: {exc}") from exc
    root = _mapping(raw, "Prometheus monitor state")
    if root.get("schemaVersion") != 1:
        raise ValueError("unsupported Prometheus monitor state schemaVersion")
    if root.get("sourceFingerprint") != source_fingerprint:
        raise ValueError(
            "Prometheus monitor data-source configuration changed; move or "
            "delete the old state file before restarting"
        )
    if root.get("baselineSha256") != baseline_sha256_value:
        raise ValueError(
            "Prometheus baseline changed; move or delete the old state file "
            "before restarting"
        )
    raw_buffer = _mapping(root.get("inferenceBuffer", {}), "inferenceBuffer")
    buffer: dict[str, TimeSeries] = {}
    for metric in metrics:
        series_raw = _mapping(raw_buffer.get(metric, {}), f"buffer.{metric}")
        timestamps = series_raw.get("timestamps", [])
        values = series_raw.get("values", [])
        if not isinstance(timestamps, list) or not isinstance(values, list):
            raise TypeError(f"buffer for {metric!r} must contain arrays")
        series = TimeSeries(
            timestamps=[float(item) for item in timestamps],
            values=[float(item) for item in values],
        )
        if len(series.timestamps) != len(series.values):
            raise ValueError(f"buffer for {metric!r} has mismatched arrays")
        buffer[metric] = series
    last_timestamp_raw = root.get("lastTimestamp")
    last_timestamp = None if last_timestamp_raw is None else float(last_timestamp_raw)
    if last_timestamp is not None and not math.isfinite(last_timestamp):
        raise ValueError("state lastTimestamp must be finite")
    last_step_raw = root.get("lastGlobalStep")
    if last_step_raw is not None and (
        isinstance(last_step_raw, bool) or not isinstance(last_step_raw, int)
    ):
        raise ValueError("state lastGlobalStep must be an integer")
    round_id = root.get("roundId", 0)
    if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
        raise ValueError("state roundId must be a non-negative integer")
    active_raw = _mapping(root.get("activeEventIds", {}), "activeEventIds")
    active = {
        _required_text(metric, "active event metric"): _required_text(
            event_id,
            f"activeEventIds.{metric}",
        )
        for metric, event_id in active_raw.items()
    }
    return PrometheusMonitorState(
        last_timestamp=last_timestamp,
        last_global_step=last_step_raw,
        inference_buffer=_trim_buffer(buffer, window_points),
        round_id=round_id,
        active_event_ids=active,
    )


def _event_id(
    task_id: str,
    target: str,
    baseline_id: str,
    abnormal_range: Mapping[str, Any],
) -> str:
    payload = {
        "taskId": task_id,
        "target": target,
        "baselineId": baseline_id,
        "startTime": abnormal_range.get("startTime"),
        "abnormalType": abnormal_range.get("abnormalType"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _top_associations(result: Mapping[str, Any], target: str) -> list[Any]:
    analysis = result.get("associationAnalysis")
    if not isinstance(analysis, Mapping):
        return []
    targets = analysis.get("targets")
    if not isinstance(targets, Mapping):
        return []
    target_result = targets.get(target)
    if not isinstance(target_result, Mapping):
        return []
    events = target_result.get("events")
    if not isinstance(events, list) or not events:
        return []
    latest = events[-1]
    if not isinstance(latest, Mapping):
        return []
    associations = latest.get("topAssociations")
    return copy.deepcopy(associations) if isinstance(associations, list) else []


def _state_change_events(
    result: Mapping[str, Any],
    *,
    targets: Sequence[str],
    task_id: str,
    baseline_id: str,
    previous_active: Mapping[str, str],
    detected_at: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    active = dict(previous_active)
    events: list[dict[str, Any]] = []
    ranges_by_metric = result.get("abnormalTimeRange", {})
    if not isinstance(ranges_by_metric, Mapping):
        ranges_by_metric = {}
    metric_errors = result.get("metricErrors", {})
    if not isinstance(metric_errors, Mapping):
        metric_errors = {}
    metric_results = result.get("results", {})
    if not isinstance(metric_results, Mapping):
        metric_results = {}
    for target in targets:
        # A failed target evaluation is not evidence of recovery. Likewise, a
        # current range waiting for history confirmation keeps an active event
        # open until the detector observes a genuinely normal window.
        if target in metric_errors or target not in ranges_by_metric:
            continue
        raw_ranges = ranges_by_metric.get(target, [])
        ranges = raw_ranges if isinstance(raw_ranges, list) else []
        target_result = metric_results.get(target, {})
        current_ranges = (
            target_result.get("currentAbnormalTimeRange", [])
            if isinstance(target_result, Mapping)
            else []
        )
        has_current_range = isinstance(current_ranges, list) and bool(current_ranges)
        if ranges and target not in active:
            latest_range = ranges[-1]
            if not isinstance(latest_range, Mapping):
                continue
            event_id = _event_id(task_id, target, baseline_id, latest_range)
            active[target] = event_id
            events.append(
                {
                    "eventId": event_id,
                    "eventType": "ABNORMAL",
                    "detectedAt": detected_at,
                    "targetMetric": target,
                    "abnormalRange": copy.deepcopy(dict(latest_range)),
                    "topAssociations": _top_associations(result, target),
                }
            )
        elif not ranges and not has_current_range and target in active:
            active_id = active.pop(target)
            recovery_raw = f"{active_id}:RECOVERED"
            recovery_id = hashlib.sha256(recovery_raw.encode("utf-8")).hexdigest()[:24]
            events.append(
                {
                    "eventId": recovery_id,
                    "eventType": "RECOVERED",
                    "detectedAt": detected_at,
                    "targetMetric": target,
                    "activeEventId": active_id,
                    "topAssociations": [],
                }
            )
    return events, active


class PrometheusMonitor:
    """Stateful three-minute poller using existing fit and detect APIs."""

    def __init__(
        self,
        config: PrometheusMonitorConfig,
        *,
        client: Any | None = None,
        baseline: BaselineModelFile | None = None,
        runner: DetectionRunner | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.policy = load_algorithm_policy_config(config.policy_config)
        self.policies = _metric_policies(self.policy, config.queries)
        self.client = client or PrometheusHttpClient(
            config.prometheus.url,
            timeout_seconds=config.prometheus.timeout_seconds,
            verify_tls=config.prometheus.verify_tls,
            bearer_token_env=config.prometheus.bearer_token_env,
            username_env=config.prometheus.username_env,
            password_env=config.prometheus.password_env,
        )
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.source_fingerprint = _source_fingerprint(config)
        self.baseline: BaselineModelFile | None = None
        self.baseline_sha256: str | None = None
        self.runner: DetectionRunner | None = None
        self.state: PrometheusMonitorState | None = None
        self._minimum_samples: dict[str, int] = {}
        if baseline is not None:
            self._activate_baseline(baseline, runner=runner)
        elif config.baseline_model.exists():
            loaded = load_baseline_model(
                config.baseline_model,
                expected_metrics=config.metrics,
            )
            self._activate_baseline(loaded, runner=runner)
        elif not config.baseline.auto_fit:
            raise ValueError(f"baseline model does not exist: {config.baseline_model}")
        elif runner is not None:
            raise ValueError("runner cannot be supplied without a baseline")

    def _activate_baseline(
        self,
        baseline: BaselineModelFile,
        *,
        runner: DetectionRunner | None = None,
    ) -> None:
        self.baseline = baseline
        self.baseline_sha256 = baseline_model_sha256(baseline)
        self.runner = runner or BaselineDetectionRunner(
            baseline,
            metrics=self.config.metrics,
            task_id=self.config.task_id,
            association_targets=self.config.target_metrics,
            metric_policies=self.policies,
            history_policy=self.policy.history_policy,
        )
        requirements = self.runner.minimum_inference_samples()
        for target in self.config.target_metrics:
            minimum = requirements.get(target)
            if (
                isinstance(minimum, bool)
                or not isinstance(minimum, int)
                or minimum <= 0
            ):
                raise ValueError(
                    f"invalid minimum inference sample count for {target!r}"
                )
            if minimum > self.config.inference_window_points:
                raise ValueError(
                    "inference_window_points is smaller than the required "
                    f"{minimum} points for target {target!r}"
                )
        self._minimum_samples = requirements
        assert self.baseline_sha256 is not None
        self.state = _load_state(
            self.config.state_file,
            metrics=self.config.metrics,
            source_fingerprint=self.source_fingerprint,
            baseline_sha256_value=self.baseline_sha256,
            window_points=self.config.inference_window_points,
        )

    def _query_window(
        self,
        start: float,
        end: float,
    ) -> tuple[TimeSeries, dict[str, TimeSeries]]:
        prom = self.config.prometheus
        global_steps = self.client.query_range(
            prom.global_step_query,
            start=start,
            end=end,
            step=prom.query_step_seconds,
        )
        metrics = self.client.fetch_range(
            self.config.queries,
            start=start,
            end=end,
            step=prom.query_step_seconds,
        )
        return global_steps, metrics

    def _bootstrap(self, now_timestamp: float) -> None:
        prom = self.config.prometheus
        start = now_timestamp - prom.lookback_seconds
        global_steps, metrics = self._query_window(start, now_timestamp)
        window = build_bootstrap_window(
            global_steps,
            metrics,
            start_step=self.config.baseline.start_step,
            end_step=self.config.baseline.end_step,
            alignment_tolerance_seconds=prom.alignment_tolerance_seconds,
        )
        expected_steps = (
            self.config.baseline.end_step - self.config.baseline.start_step + 1
        )
        coverage = len(window.observed_steps) / expected_steps
        if coverage < self.config.baseline.minimum_step_coverage_ratio:
            raise ValueError(
                "baseline global-step coverage is too low: "
                f"{coverage:.3f} < "
                f"{self.config.baseline.minimum_step_coverage_ratio:.3f}"
            )
        for metric in self.config.metrics:
            count = len(window.series.get(metric, TimeSeries()))
            if count < self.config.baseline.minimum_samples_per_metric:
                raise ValueError(
                    f"baseline has only {count} aligned samples for {metric!r}; "
                    "increase Prometheus history/coverage or lower the configured "
                    "minimum deliberately"
                )
        baseline_id = self.config.baseline.baseline_id or (
            f"{_safe_task_name(self.config.task_id)}-{self.source_fingerprint[:12]}"
        )
        model = fit_baseline_model(
            window.series,
            self.config.metrics,
            baseline_id=baseline_id,
            source_type="prometheus",
            metric_policies=self.policies,
            history_policy=self.policy.history_policy,
        )
        save_baseline_model(model, self.config.baseline_model)
        self._activate_baseline(model)
        assert self.state is not None
        self.state.last_timestamp = window.end_timestamp

    def _commit_state(self, state: PrometheusMonitorState) -> None:
        assert self.baseline_sha256 is not None
        _write_json_atomically(
            self.config.state_file,
            _state_payload(
                state,
                metrics=self.config.metrics,
                source_fingerprint=self.source_fingerprint,
                baseline_sha256=self.baseline_sha256,
            ),
        )

    def poll_once(self) -> PrometheusPollOutcome:
        now = self._now().astimezone(timezone.utc)
        now_timestamp = now.timestamp()
        try:
            if self.baseline is None:
                self._bootstrap(now_timestamp)
            assert self.baseline is not None
            assert self.runner is not None
            assert self.state is not None
            prom = self.config.prometheus
            start = (
                now_timestamp - prom.lookback_seconds
                if self.state.last_timestamp is None
                else self.state.last_timestamp - prom.overlap_seconds
            )
            global_steps, raw_metrics = self._query_window(start, now_timestamp)
            latest_step = latest_integer_step(global_steps)
            if latest_step is None:
                return PrometheusPollOutcome(
                    status="waiting_for_global_step",
                )
            if (
                self.state.last_global_step is not None
                and latest_step < self.state.last_global_step
            ):
                raise ValueError(
                    "training global step moved backwards; use a new task_id/state "
                    "file for the restarted run"
                )
            if (
                self.state.last_global_step is not None
                and latest_step == self.state.last_global_step
            ):
                candidate = PrometheusMonitorState(
                    last_timestamp=now_timestamp,
                    last_global_step=latest_step,
                    inference_buffer=_copy_buffer(self.state.inference_buffer),
                    round_id=self.state.round_id,
                    active_event_ids=dict(self.state.active_event_ids),
                )
                self._commit_state(candidate)
                self.state = candidate
                return PrometheusPollOutcome(status="no_new_data")
            new_metrics = filter_after_global_step(
                global_steps,
                raw_metrics,
                minimum_step_exclusive=self.config.baseline.end_step,
                alignment_tolerance_seconds=prom.alignment_tolerance_seconds,
            )
            merged = {
                metric: _merge_series(
                    self.state.inference_buffer.get(metric, TimeSeries()),
                    new_metrics.get(metric, TimeSeries()),
                )
                for metric in self.config.metrics
            }
            buffer = _trim_buffer(merged, self.config.inference_window_points)
            candidate = PrometheusMonitorState(
                last_timestamp=now_timestamp,
                last_global_step=latest_step,
                inference_buffer=buffer,
                round_id=self.state.round_id,
                active_event_ids=dict(self.state.active_event_ids),
            )
            has_points = any(len(series) for series in new_metrics.values())
            enough = all(
                len(buffer.get(target, TimeSeries())) >= self._minimum_samples[target]
                for target in self.config.target_metrics
            )
            if not has_points or not enough:
                self._commit_state(candidate)
                self.state = candidate
                return PrometheusPollOutcome(
                    status="waiting_for_data" if has_points else "no_new_data"
                )

            runtime_snapshot = self.runner.snapshot_runtime_state()
            try:
                result = self.runner.run(_copy_buffer(buffer))
                detected_at = now.isoformat()
                events, active = _state_change_events(
                    result,
                    targets=self.config.target_metrics,
                    task_id=self.config.task_id,
                    baseline_id=self.baseline.baseline_id,
                    previous_active=self.state.active_event_ids,
                    detected_at=detected_at,
                )
                candidate.round_id = self.state.round_id + 1
                candidate.active_event_ids = active
                result["events"] = events
                result["metadata"] = {
                    "schemaVersion": 1,
                    "mode": "prometheus-polling",
                    "sourceType": "prometheus",
                    "roundId": candidate.round_id,
                    "detectedAt": detected_at,
                    "baselineId": self.baseline.baseline_id,
                    "baselineSha256": self.baseline_sha256,
                    "sourceFingerprint": self.source_fingerprint,
                    "queryRange": {"start": start, "end": now_timestamp},
                    "queryStepSeconds": prom.query_step_seconds,
                    "pollIntervalSeconds": self.config.poll_interval_seconds,
                    "lastGlobalStep": latest_step,
                    "selectedMetrics": list(self.config.metrics),
                    "targetMetrics": list(self.config.target_metrics),
                    "candidateMetrics": list(self.config.candidate_metrics),
                    "perMetricPointCounts": {
                        metric: len(buffer.get(metric, TimeSeries()))
                        for metric in self.config.metrics
                    },
                }
                latest_path = _write_json_atomically(
                    self.config.output_dir
                    / f"{_safe_task_name(self.config.task_id)}_latest.json",
                    result,
                )
                event_path: Path | None = None
                if events:
                    event_key = hashlib.sha256(
                        ":".join(event["eventId"] for event in events).encode("utf-8")
                    ).hexdigest()[:24]
                    event_path = _write_json_atomically(
                        self.config.output_dir / f"event_{event_key}.json",
                        result,
                    )
                self._commit_state(candidate)
            except KeyboardInterrupt:
                self.runner.restore_runtime_state(runtime_snapshot)
                raise
            except Exception:
                self.runner.restore_runtime_state(runtime_snapshot)
                raise
            self.state = candidate
            return PrometheusPollOutcome(
                status="state_changed" if events else "detected",
                result=result,
                output_path=latest_path,
                event_path=event_path,
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad poll must not stop service.
            return PrometheusPollOutcome(
                status="error",
                error={"type": type(exc).__name__, "message": str(exc)},
            )

    def run_forever(
        self,
        *,
        max_polls: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        polls = 0
        while max_polls is None or polls < max_polls:
            outcome = self.poll_once()
            polls += 1
            if outcome.status == "error":
                assert outcome.error is not None
                sys.stderr.write(
                    f"poll error: {outcome.error['type']}: {outcome.error['message']}\n"
                )
            elif outcome.event_path is not None:
                sys.stdout.write(f"event JSON: {outcome.event_path}\n")
            elif outcome.output_path is not None:
                sys.stdout.write(f"latest result: {outcome.output_path}\n")
            if max_polls is None or polls < max_polls:
                sleep(self.config.poll_interval_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Poll Prometheus for RL-Insight degradation detection",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--once",
        action="store_true",
        help="perform one bootstrap/poll cycle and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        monitor = PrometheusMonitor(load_prometheus_monitor_config(args.config))
        if args.once:
            outcome = monitor.poll_once()
            if outcome.status == "error":
                assert outcome.error is not None
                raise ValueError(f"{outcome.error['type']}: {outcome.error['message']}")
            sys.stdout.write(f"status: {outcome.status}\n")
            if outcome.output_path is not None:
                sys.stdout.write(f"latest result: {outcome.output_path}\n")
            if outcome.event_path is not None:
                sys.stdout.write(f"event JSON: {outcome.event_path}\n")
        else:
            monitor.run_forever()
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, TypeError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BaselineBootstrapConfig",
    "PrometheusConnectionConfig",
    "PrometheusMonitor",
    "PrometheusMonitorConfig",
    "PrometheusMonitorState",
    "PrometheusPollOutcome",
    "TargetSelectionConfig",
    "build_parser",
    "load_prometheus_monitor_config",
    "main",
]
