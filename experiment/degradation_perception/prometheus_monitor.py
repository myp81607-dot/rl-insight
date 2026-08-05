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

"""Minimal Prometheus discovery, training, detection, and analysis CLI."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from .algorithm_policy import (
    DEFAULT_ALGORITHM_POLICY_PATH,
    AlgorithmPolicyConfig,
    load_algorithm_policy_config,
)
from .baseline_model import (
    BaselineDetectionRunner,
    BaselineModelFile,
    MetricBaseline,
    baseline_model_from_dict,
    fit_baseline_model,
)
from .perception_config import TimeSeries
from .policy import resolve_metric_policy
from .prometheus_source import (
    PrometheusHttpClient,
    PrometheusSeries,
    build_bootstrap_window,
    filter_after_global_step,
    latest_integer_step,
)
from .serialization import to_json_serializable


TRAINING_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive number")
    return number


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _patterns(value: Any, name: str, default: Sequence[str]) -> tuple[str, ...]:
    raw = list(default) if value is None else value
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(_text(item, name).casefold() for item in raw)
    return tuple(dict.fromkeys(result))


def _path(value: Any, name: str, *, base: Path) -> Path:
    path = Path(_text(value, name)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


@dataclass(frozen=True)
class MonitorConfig:
    """Small YAML-owned surface for the experiment runtime."""

    prometheus_url: str
    series_selector: str
    query_step_seconds: float
    lookback_seconds: float
    alignment_tolerance_seconds: float
    timeout_seconds: float
    verify_tls: bool
    baseline_start_step: int
    baseline_end_step: int
    minimum_baseline_samples: int
    inference_points: int
    poll_interval_seconds: float
    top_k: int
    target_patterns: tuple[str, ...]
    global_step_patterns: tuple[str, ...]
    algorithm_config: Path
    output_dir: Path

    @property
    def query_file(self) -> Path:
        return self.output_dir / "query.json"

    @property
    def training_file(self) -> Path:
        return self.output_dir / "training.json"

    @property
    def result_file(self) -> Path:
        return self.output_dir / "anomalies.json"


def load_config(path: str | Path) -> MonitorConfig:
    """Load the one intentionally small runtime YAML file."""

    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"could not load config {source}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("config root must be an object")
    base = source.parent

    baseline_steps = raw.get("baseline_steps", [5, 25])
    if (
        not isinstance(baseline_steps, list)
        or len(baseline_steps) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in baseline_steps)
    ):
        raise ValueError("baseline_steps must be [start_step, end_step]")
    start_step, end_step = baseline_steps
    if start_step > end_step:
        raise ValueError("baseline_steps start must not exceed end")
    verify_tls = raw.get("verify_tls", True)
    if not isinstance(verify_tls, bool):
        raise ValueError("verify_tls must be true or false")

    return MonitorConfig(
        prometheus_url=_text(
            raw.get("prometheus_url", "http://127.0.0.1:9090"),
            "prometheus_url",
        ).rstrip("/"),
        series_selector=_text(
            raw.get("series_selector", '{__name__=~".+"}'),
            "series_selector",
        ),
        query_step_seconds=_positive_float(
            raw.get("query_step_seconds", 10), "query_step_seconds"
        ),
        lookback_seconds=_positive_float(
            raw.get("lookback_seconds", 86400), "lookback_seconds"
        ),
        alignment_tolerance_seconds=_positive_float(
            raw.get("alignment_tolerance_seconds", 6),
            "alignment_tolerance_seconds",
        ),
        timeout_seconds=_positive_float(
            raw.get("timeout_seconds", 30), "timeout_seconds"
        ),
        verify_tls=verify_tls,
        baseline_start_step=start_step,
        baseline_end_step=end_step,
        minimum_baseline_samples=_positive_int(
            raw.get("minimum_baseline_samples", 10),
            "minimum_baseline_samples",
        ),
        inference_points=_positive_int(
            raw.get("inference_points", 120), "inference_points"
        ),
        poll_interval_seconds=_positive_float(
            raw.get("poll_interval_seconds", 180), "poll_interval_seconds"
        ),
        top_k=_positive_int(raw.get("top_k", 5), "top_k"),
        target_patterns=_patterns(
            raw.get("target_patterns"), "target_patterns", ("time", "timing")
        ),
        global_step_patterns=_patterns(
            raw.get("global_step_patterns"),
            "global_step_patterns",
            ("global_step",),
        ),
        algorithm_config=_path(
            raw.get("algorithm_config", str(DEFAULT_ALGORITHM_POLICY_PATH)),
            "algorithm_config",
            base=base,
        ),
        output_dir=_path(raw.get("output_dir", "./output"), "output_dir", base=base),
    )


@dataclass(frozen=True)
class SeriesSpec:
    """Concrete Prometheus series plus its algorithm role."""

    series: PrometheusSeries
    role: str

    @property
    def identifier(self) -> str:
        return self.series.identifier

    def to_dict(self) -> dict[str, Any]:
        result = self.series.to_dict()
        result["role"] = self.role
        return result

    @classmethod
    def from_dict(cls, raw: Any) -> "SeriesSpec":
        if not isinstance(raw, Mapping):
            raise ValueError("training series entry must be an object")
        metric_name = _text(raw.get("metricName"), "series.metricName")
        labels_raw = raw.get("labels", {})
        if not isinstance(labels_raw, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in labels_raw.items()
        ):
            raise ValueError("series.labels must be string pairs")
        role = _text(raw.get("role"), "series.role")
        if role not in {"global_step", "target", "candidate"}:
            raise ValueError(f"unsupported series role: {role!r}")
        return cls(
            PrometheusSeries(metric_name, dict(labels_raw)),
            role,
        )


def _role(series: PrometheusSeries, config: MonitorConfig) -> str:
    name = series.metric_name.casefold()
    if any(pattern in name for pattern in config.global_step_patterns):
        return "global_step"
    if any(pattern in name for pattern in config.target_patterns):
        return "target"
    return "candidate"


def _metric_policies(
    policy: AlgorithmPolicyConfig,
    specs: Sequence[SeriesSpec],
    *,
    top_k: int,
) -> dict[str, dict[str, Any]]:
    policies = policy.for_metrics([spec.identifier for spec in specs])
    for spec in specs:
        metric_policy = policies[spec.identifier]
        if spec.role == "candidate":
            metric_policy["abnormal_type"] = "BOTH"
        association = copy.deepcopy(metric_policy.get("association", {}))
        association["top_k"] = top_k
        metric_policy["association"] = association
    return policies


def _write_json(path: Path, payload: Any) -> Path:
    serialized = json.dumps(
        to_json_serializable(payload),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = handle.name
        os.replace(temporary, path)
        temporary = None
        return path
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load {name} {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{name} root must be an object")
    return raw


def _trim(series: TimeSeries, points: int) -> TimeSeries:
    if len(series) <= points:
        return series
    return TimeSeries(series.timestamps[-points:], series.values[-points:])


class PrometheusRuntime:
    """One direct path from Prometheus JSON to the existing algorithm."""

    def __init__(
        self,
        config: MonitorConfig,
        *,
        client: Any | None = None,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.config = config
        self.client = client or PrometheusHttpClient(
            config.prometheus_url,
            timeout_seconds=config.timeout_seconds,
            verify_tls=config.verify_tls,
        )
        self.now = now

    def _range(self) -> tuple[float, float]:
        end = self.now().astimezone(timezone.utc).timestamp()
        return end - self.config.lookback_seconds, end

    def _snapshot(
        self,
    ) -> tuple[
        list[SeriesSpec],
        dict[str, TimeSeries],
        list[dict[str, str]],
        float,
        float,
    ]:
        start, end = self._range()
        discovered = self.client.list_series(
            self.config.series_selector,
            start=start,
            end=end,
        )
        specs = [SeriesSpec(series, _role(series, self.config)) for series in discovered]
        values: dict[str, TimeSeries] = {}
        failures: list[dict[str, str]] = []
        for spec in specs:
            try:
                values[spec.identifier] = self.client.query_range(
                    spec.series.selector,
                    start=start,
                    end=end,
                    step=self.config.query_step_seconds,
                )
            except Exception as exc:  # One bad series must not hide all others.
                failures.append(
                    {
                        "id": spec.identifier,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                values[spec.identifier] = TimeSeries()
        return specs, values, failures, start, end

    @staticmethod
    def _select_global_step(
        specs: Sequence[SeriesSpec], values: Mapping[str, TimeSeries]
    ) -> SeriesSpec:
        candidates: list[tuple[int, int, str, SeriesSpec]] = []
        for spec in specs:
            if spec.role != "global_step":
                continue
            series = values.get(spec.identifier, TimeSeries())
            latest = latest_integer_step(series)
            if latest is not None:
                candidates.append((latest, len(series), spec.identifier, spec))
        if not candidates:
            raise ValueError(
                "no readable global-step series was found; adjust global_step_patterns"
            )
        return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]

    def query(self) -> dict[str, Any]:
        specs, values, failures, start, end = self._snapshot()
        payload = {
            "schemaVersion": 1,
            "queriedAt": self.now().astimezone(timezone.utc).isoformat(),
            "prometheusUrl": self.config.prometheus_url,
            "queryRange": {"start": start, "end": end},
            "series": [
                {
                    **spec.to_dict(),
                    "samples": {
                        "timestamps": values[spec.identifier].timestamps,
                        "values": values[spec.identifier].values,
                    },
                }
                for spec in specs
            ],
            "failures": failures,
            "summary": {
                "series": len(specs),
                "globalSteps": sum(spec.role == "global_step" for spec in specs),
                "targets": sum(spec.role == "target" for spec in specs),
                "candidates": sum(spec.role == "candidate" for spec in specs),
                "failedQueries": len(failures),
            },
        }
        _write_json(self.config.query_file, payload)
        return payload

    def train(self) -> dict[str, Any]:
        specs, values, failures, start, end = self._snapshot()
        global_spec = self._select_global_step(specs, values)
        algorithm_specs = [
            spec for spec in specs if spec.role in {"target", "candidate"}
        ]
        window = build_bootstrap_window(
            values[global_spec.identifier],
            {spec.identifier: values[spec.identifier] for spec in algorithm_specs},
            start_step=self.config.baseline_start_step,
            end_step=self.config.baseline_end_step,
            alignment_tolerance_seconds=self.config.alignment_tolerance_seconds,
        )
        policy = load_algorithm_policy_config(self.config.algorithm_config)
        policies = _metric_policies(policy, algorithm_specs, top_k=self.config.top_k)
        fitted: dict[str, MetricBaseline] = {}
        retained: list[SeriesSpec] = []
        skipped: list[dict[str, Any]] = []
        created_at = self.now().astimezone(timezone.utc).isoformat()

        for spec in algorithm_specs:
            series = window.series.get(spec.identifier, TimeSeries())
            required = max(
                self.config.minimum_baseline_samples,
                int(
                    resolve_metric_policy(
                        spec.identifier, policies[spec.identifier]
                    )["minimum_standard_points"]
                ),
            )
            if len(series) < required:
                skipped.append(
                    {
                        **spec.to_dict(),
                        "reason": "insufficient_baseline_samples",
                        "samples": len(series),
                        "required": required,
                    }
                )
                continue
            try:
                single = fit_baseline_model(
                    {spec.identifier: series},
                    [spec.identifier],
                    baseline_id="prometheus",
                    source_type="prometheus",
                    metric_policies={spec.identifier: policies[spec.identifier]},
                    history_policy=policy.history_policy,
                    created_at=created_at,
                )
            except Exception as exc:
                skipped.append(
                    {
                        **spec.to_dict(),
                        "reason": "baseline_fit_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            fitted[spec.identifier] = single.metrics[spec.identifier]
            retained.append(spec)

        targets = [spec for spec in retained if spec.role == "target"]
        if not targets:
            raise ValueError(
                "no time/timing target could be trained; inspect skippedSeries"
            )
        baseline = baseline_model_from_dict(
            BaselineModelFile(
                baseline_id="prometheus",
                created_at=created_at,
                source_type="prometheus",
                metrics=fitted,
                history_policy=policy.history_policy,
            ).to_dict()
        )
        payload = {
            "schemaVersion": TRAINING_SCHEMA_VERSION,
            "createdAt": created_at,
            "prometheusUrl": self.config.prometheus_url,
            "seriesSelector": self.config.series_selector,
            "queryRange": {"start": start, "end": end},
            "trainingInterval": {
                "startStep": self.config.baseline_start_step,
                "endStep": self.config.baseline_end_step,
                "observedSteps": list(window.observed_steps),
                "sourceStartTime": window.start_timestamp,
                "sourceEndTime": window.end_timestamp,
            },
            "globalStepSeries": global_spec.to_dict(),
            "series": [spec.to_dict() for spec in retained],
            "skippedSeries": skipped,
            "queryFailures": failures,
            "model": baseline.to_dict(),
            "summary": {
                "discovered": len(specs),
                "trained": len(retained),
                "targets": len(targets),
                "candidates": sum(spec.role == "candidate" for spec in retained),
                "skipped": len(skipped),
            },
        }
        _write_json(self.config.training_file, payload)
        return payload

    def _load_training(
        self,
    ) -> tuple[dict[str, Any], BaselineModelFile, SeriesSpec, list[SeriesSpec]]:
        artifact = _read_json(self.config.training_file, "training artifact")
        if artifact.get("schemaVersion") != TRAINING_SCHEMA_VERSION:
            raise ValueError("unsupported training artifact schemaVersion")
        baseline = baseline_model_from_dict(artifact.get("model"))
        global_spec = SeriesSpec.from_dict(artifact.get("globalStepSeries"))
        raw_specs = artifact.get("series")
        if not isinstance(raw_specs, list) or not raw_specs:
            raise ValueError("training artifact series must be a non-empty array")
        specs = [SeriesSpec.from_dict(item) for item in raw_specs]
        return artifact, baseline, global_spec, specs

    def detect(self) -> dict[str, Any]:
        artifact, baseline, global_spec, specs = self._load_training()
        start, end = self._range()
        failures: list[dict[str, str]] = []

        try:
            global_steps = self.client.query_range(
                global_spec.series.selector,
                start=start,
                end=end,
                step=self.config.query_step_seconds,
            )
        except Exception as exc:
            raise ValueError(f"could not query global step: {exc}") from exc

        raw_values: dict[str, TimeSeries] = {}
        for spec in specs:
            try:
                raw_values[spec.identifier] = self.client.query_range(
                    spec.series.selector,
                    start=start,
                    end=end,
                    step=self.config.query_step_seconds,
                )
            except Exception as exc:
                raw_values[spec.identifier] = TimeSeries()
                failures.append(
                    {
                        "id": spec.identifier,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        interval = artifact.get("trainingInterval", {})
        if not isinstance(interval, Mapping):
            raise ValueError("trainingInterval must be an object")
        baseline_end_step = interval.get("endStep")
        if isinstance(baseline_end_step, bool) or not isinstance(
            baseline_end_step, int
        ):
            raise ValueError("trainingInterval.endStep must be an integer")
        inference = filter_after_global_step(
            global_steps,
            raw_values,
            minimum_step_exclusive=baseline_end_step,
            alignment_tolerance_seconds=self.config.alignment_tolerance_seconds,
        )
        inference = {
            metric: _trim(series, self.config.inference_points)
            for metric, series in inference.items()
        }

        targets = [spec.identifier for spec in specs if spec.role == "target"]
        policy = load_algorithm_policy_config(self.config.algorithm_config)
        policies = _metric_policies(policy, specs, top_k=self.config.top_k)
        runner = BaselineDetectionRunner(
            baseline,
            metrics=[spec.identifier for spec in specs],
            task_id="prometheus",
            association_targets=targets,
            metric_policies=policies,
            history_policy=policy.history_policy,
        )
        result = runner.run(inference)
        payload = self._compact_result(
            result,
            specs,
            latest_step=latest_integer_step(global_steps),
            failures=failures,
            point_counts={metric: len(series) for metric, series in inference.items()},
        )
        _write_json(self.config.result_file, payload)
        return payload

    def _compact_result(
        self,
        result: Mapping[str, Any],
        specs: Sequence[SeriesSpec],
        *,
        latest_step: int | None,
        failures: list[dict[str, str]],
        point_counts: Mapping[str, int],
    ) -> dict[str, Any]:
        metadata = {spec.identifier: spec for spec in specs}
        ranges = result.get("abnormalTimeRange", {})
        if not isinstance(ranges, Mapping):
            ranges = {}
        states = result.get("states", {})
        if not isinstance(states, Mapping):
            states = {}
        analysis = result.get("associationAnalysis", {})
        analysis_targets = analysis.get("targets", {}) if isinstance(analysis, Mapping) else {}
        if not isinstance(analysis_targets, Mapping):
            analysis_targets = {}

        target_results: list[dict[str, Any]] = []
        anomalies: list[dict[str, Any]] = []
        for spec in specs:
            raw_intervals = ranges.get(spec.identifier, [])
            intervals = raw_intervals if isinstance(raw_intervals, list) else []
            if intervals:
                anomalies.append(
                    {
                        **spec.to_dict(),
                        "intervals": intervals,
                    }
                )
            if spec.role != "target":
                continue
            target_analysis = analysis_targets.get(spec.identifier, {})
            events = (
                target_analysis.get("events", [])
                if isinstance(target_analysis, Mapping)
                else []
            )
            compact_events: list[dict[str, Any]] = []
            if isinstance(events, list):
                for event in events:
                    if not isinstance(event, Mapping):
                        continue
                    top = event.get("topAssociations", [])
                    top5: list[dict[str, Any]] = []
                    if isinstance(top, list):
                        for item in top[: self.config.top_k]:
                            if not isinstance(item, Mapping):
                                continue
                            candidate_id = str(item.get("metric", ""))
                            candidate = metadata.get(candidate_id)
                            top5.append(
                                {
                                    "rank": item.get("rank"),
                                    "series": candidate_id,
                                    "metricName": (
                                        candidate.series.metric_name
                                        if candidate is not None
                                        else candidate_id
                                    ),
                                    "labels": (
                                        dict(candidate.series.labels)
                                        if candidate is not None
                                        else {}
                                    ),
                                    "abnormalContributionPercent": item.get(
                                        "abnormalContribution", 0.0
                                    ),
                                }
                            )
                    compact_events.append(
                        {
                            "interval": event.get("targetAbnormalRange"),
                            "top5": top5,
                        }
                    )
            state = states.get(spec.identifier)
            target_results.append(
                {
                    **spec.to_dict(),
                    "status": (
                        "abnormal"
                        if intervals
                        else "normal"
                        if state == 0
                        else "insufficient_data"
                    ),
                    "state": state,
                    "samples": point_counts.get(spec.identifier, 0),
                    "intervals": intervals,
                    "events": compact_events,
                    "top5": compact_events[-1]["top5"] if compact_events else [],
                }
            )

        metric_errors = result.get("metricErrors", {})
        overall_status = (
            "abnormal"
            if anomalies
            else "normal"
            if any(target["status"] == "normal" for target in target_results)
            else "insufficient_data"
        )
        payload = {
            "schemaVersion": RESULT_SCHEMA_VERSION,
            "detectedAt": self.now().astimezone(timezone.utc).isoformat(),
            "status": overall_status,
            "lastGlobalStep": latest_step,
            "targets": target_results,
            "anomalousMetrics": anomalies,
            "queryFailures": failures,
            "metricErrors": metric_errors if isinstance(metric_errors, Mapping) else {},
            "summary": {
                "targets": len(target_results),
                "abnormalTargets": sum(
                    target["status"] == "abnormal" for target in target_results
                ),
                "anomalousMetrics": len(anomalies),
                "queryFailures": len(failures),
            },
        }
        return to_json_serializable(payload)

    def start(self, *, sleep: Callable[[float], None] = time.sleep) -> None:
        """Poll forever; association is already included in every detection."""

        while True:
            try:
                result = self.detect()
                print(
                    json.dumps(
                        {
                            "status": result["status"],
                            "lastGlobalStep": result["lastGlobalStep"],
                            "result": str(self.config.result_file),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # A later poll can recover from transient errors.
                print(f"poll error: {type(exc).__name__}: {exc}", file=sys.stderr)
            sleep(self.config.poll_interval_seconds)


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _name_tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", value.casefold()) if token]


def _token_subsequence(needle: str, candidate: str) -> bool:
    """Allow ``time_step`` to find names such as ``timing_s_step``."""

    wanted = _name_tokens(needle)
    available = _name_tokens(candidate)
    if not wanted:
        return False
    position = 0
    for token in wanted:
        stem = token[:3]
        while position < len(available) and not available[position].startswith(stem):
            position += 1
        if position == len(available):
            return False
        position += 1
    return True


def select_analysis(result: Mapping[str, Any], target: str | None) -> dict[str, Any]:
    """Return one target's latest Top-5 view, or all targets when omitted."""

    raw_targets = result.get("targets", [])
    if not isinstance(raw_targets, list):
        raise ValueError("anomaly result targets must be an array")
    targets = [item for item in raw_targets if isinstance(item, Mapping)]
    selected = targets
    if target is not None:
        needle = _normalized(_text(target, "target"))
        exact = [
            item
            for item in targets
            if needle
            in {
                _normalized(str(item.get("id", ""))),
                _normalized(str(item.get("metricName", ""))),
            }
        ]
        selected = exact or [
            item
            for item in targets
            if needle in _normalized(str(item.get("id", "")))
            or needle in _normalized(str(item.get("metricName", "")))
            or _token_subsequence(target, str(item.get("metricName", "")))
        ]
        if not selected:
            available = ", ".join(str(item.get("metricName")) for item in targets)
            raise ValueError(f"target {target!r} was not found; available: {available}")
        if len(selected) > 1:
            names = ", ".join(str(item.get("id")) for item in selected[:10])
            raise ValueError(f"target {target!r} matches multiple series: {names}")

    views = [
        {
            "target": item.get("id"),
            "metricName": item.get("metricName"),
            "labels": item.get("labels", {}),
            "status": item.get("status"),
            "intervals": item.get("intervals", []),
            "top5": item.get("top5", []),
        }
        for item in selected
    ]
    return {"targets": views}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiment.degradation_perception",
        description="Prometheus degradation experiment",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("start", "poll forever"),
        ("train", "fit step-range baseline and write training.json"),
        ("query", "download all discovered series to query.json"),
        ("detect", "run one detection and write anomalies.json"),
        ("analyze", "show a target's latest Top-5 associations"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--config",
            type=Path,
            default=Path(__file__).with_name("config.yaml"),
        )
        if name == "analyze":
            command.add_argument("--target", help="metric name or unique substring")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(list(argv) if argv is not None else None)
    if unknown:
        if (
            args.command == "analyze"
            and args.target is None
            and len(unknown) == 1
            and unknown[0].startswith("--")
            and len(unknown[0]) > 2
        ):
            args.target = unknown[0][2:]
        else:
            parser.error("unrecognized arguments: " + " ".join(unknown))
    try:
        config = load_config(args.config)
        runtime = PrometheusRuntime(config)
        if args.command == "query":
            payload = runtime.query()
            print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
            print(f"saved: {config.query_file}")
        elif args.command == "train":
            payload = runtime.train()
            print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
            print(f"saved: {config.training_file}")
        elif args.command == "detect":
            payload = runtime.detect()
            print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
            print(f"saved: {config.result_file}")
        elif args.command == "analyze":
            payload = _read_json(config.result_file, "anomaly result")
            print(
                json.dumps(
                    select_analysis(payload, args.target),
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                )
            )
        else:
            runtime.start()
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MonitorConfig",
    "PrometheusRuntime",
    "SeriesSpec",
    "build_parser",
    "load_config",
    "main",
    "select_analysis",
]
