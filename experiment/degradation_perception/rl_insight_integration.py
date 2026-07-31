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

"""Formal bridge from RL-Insight monitoring to degradation perception."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rl_insight.server.http_api import get_server_services, server_url
from rl_insight.server.network import (
    format_host_port,
    local_addresses,
    service_url_from_server_url,
)
from rl_insight.utils.prometheus_utils import (
    MetricRegistry,
    start_metrics_http_server,
    update_prometheus_config,
)

from .algorithm import DegradationPerception
from .prometheus_matrix_adapter import convert_matrix_response
from .prometheus_query import (
    PrometheusQueryClient,
    PrometheusQueryError,
    validate_prometheus_endpoint,
)


DEFAULT_BASELINE_WINDOW_SECONDS = 30 * 60
DEFAULT_INFERENCE_WINDOW_SECONDS = 30 * 60
DEFAULT_QUERY_STEP_SECONDS = 10.0
DEGRADATION_SCRAPE_JOB = "rl_insight_degradation"
_METRIC_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class RLInsightIntegrationError(ValueError):
    """Configuration or discovery failure at the integration boundary."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = str(code)
        self.message = str(message)
        self.details = dict(details or {})
        super().__init__(f"{self.code}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class DetectionWindow:
    start: float
    end: float

    def to_dict(self) -> dict[str, float]:
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class DetectionWindows:
    standard: DetectionWindow
    inference: DetectionWindow
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "standard": self.standard.to_dict(),
            "inference": self.inference.to_dict(),
            "source": self.source,
        }


@dataclass(frozen=True)
class MetricQuerySpec:
    """Logical detector metric plus its Prometheus query representation."""

    logical_name: str
    prometheus_metric: str
    query_template: str | None = None
    series_policy: str = "exactly_one"
    select_labels: Mapping[str, str] | None = None

    def query_for(self, labels: Mapping[str, str]) -> str:
        selector = _label_selector(labels)
        if self.query_template is None:
            if not _METRIC_NAME.fullmatch(self.prometheus_metric):
                raise RLInsightIntegrationError(
                    "invalid_metric_name",
                    f"invalid Prometheus metric name: {self.prometheus_metric!r}",
                )
            return f"{self.prometheus_metric}{selector}"
        query = str(self.query_template).strip()
        if not query:
            raise RLInsightIntegrationError(
                "invalid_query_template",
                f"query template for {self.logical_name!r} is empty",
            )
        return query.replace("{{metric}}", self.prometheus_metric).replace(
            "{{labels}}",
            selector,
        )


def resolve_prometheus_endpoint(
    *,
    override: str | None = None,
    rl_insight_server_url: str | None = None,
    service_fetcher: Callable[[], Mapping[str, Any]] = get_server_services,
) -> str:
    """Resolve the managed Prometheus query endpoint from public service data.

    ``override`` exists for tests and development. Production discovery uses
    the RL-Insight server URL plus ``GET /api/v1/services``.
    """

    if override:
        return validate_prometheus_endpoint(override)
    base_url = str(rl_insight_server_url or server_url()).strip().rstrip("/")
    if not base_url:
        raise RLInsightIntegrationError(
            "rl_insight_server_url_missing",
            "set RL_INSIGHT_SERVER_URL so degradation can discover Prometheus",
        )
    services = service_fetcher()
    if not isinstance(services, Mapping) or services.get("status") != "ok":
        raise RLInsightIntegrationError(
            "service_discovery_failed",
            "RL-Insight Server did not return usable service information",
        )
    explicit_endpoint = services.get("prometheus_url") or services.get(
        "prometheus_endpoint"
    )
    if explicit_endpoint:
        return validate_prometheus_endpoint(str(explicit_endpoint))
    endpoint = service_url_from_server_url(
        base_url,
        services.get("prometheus_port"),
    )
    if not endpoint:
        raise RLInsightIntegrationError(
            "prometheus_service_unavailable",
            "RL-Insight Server reports no enabled Prometheus service",
        )
    return validate_prometheus_endpoint(endpoint)


def derive_detection_windows(
    *,
    now: float | None = None,
    standard_start: float | None = None,
    standard_end: float | None = None,
    inference_start: float | None = None,
    inference_end: float | None = None,
    task_metadata: Mapping[str, Any] | None = None,
    baseline_window_seconds: float = DEFAULT_BASELINE_WINDOW_SECONDS,
    inference_window_seconds: float = DEFAULT_INFERENCE_WINDOW_SECONDS,
) -> DetectionWindows:
    """Resolve explicit windows, task metadata, then duration defaults."""

    baseline_duration = _positive_finite(
        baseline_window_seconds,
        "baseline_window_seconds",
    )
    inference_duration = _positive_finite(
        inference_window_seconds,
        "inference_window_seconds",
    )
    current = float(time.time() if now is None else now)
    if not math.isfinite(current):
        raise RLInsightIntegrationError(
            "invalid_detection_time",
            "current detection time must be finite",
        )
    metadata = dict(task_metadata or {})
    task_start = _optional_timestamp(
        metadata.get("start_time", metadata.get("startTime")),
        "task metadata start_time",
    )
    explicit = any(
        value is not None
        for value in (
            standard_start,
            standard_end,
            inference_start,
            inference_end,
        )
    )

    resolved_inference_end = _optional_timestamp(
        inference_end,
        "inference_end",
    )
    if resolved_inference_end is None:
        resolved_inference_end = current
    resolved_inference_start = _optional_timestamp(
        inference_start,
        "inference_start",
    )
    if resolved_inference_start is None:
        resolved_inference_start = resolved_inference_end - inference_duration
        if task_start is not None and task_start < resolved_inference_end:
            resolved_inference_start = max(
                task_start,
                resolved_inference_start,
            )

    resolved_standard_end = _optional_timestamp(
        standard_end,
        "standard_end",
    )
    if resolved_standard_end is None:
        resolved_standard_end = resolved_inference_start
    resolved_standard_start = _optional_timestamp(
        standard_start,
        "standard_start",
    )
    if resolved_standard_start is None:
        resolved_standard_start = resolved_standard_end - baseline_duration
        if task_start is not None and task_start < resolved_standard_end:
            resolved_standard_start = max(
                task_start,
                resolved_standard_start,
            )

    if resolved_standard_start >= resolved_standard_end:
        raise RLInsightIntegrationError(
            "invalid_standard_window",
            "standard window start must be earlier than end",
        )
    if resolved_inference_start >= resolved_inference_end:
        raise RLInsightIntegrationError(
            "invalid_inference_window",
            "inference window start must be earlier than end",
        )
    source = "explicit_cli_or_api" if explicit else (
        "task_metadata" if task_start is not None else "degradation_defaults"
    )
    return DetectionWindows(
        standard=DetectionWindow(
            resolved_standard_start,
            resolved_standard_end,
        ),
        inference=DetectionWindow(
            resolved_inference_start,
            resolved_inference_end,
        ),
        source=source,
    )


class PrometheusDegradationRunner:
    """Query managed Prometheus and invoke the unchanged detector."""

    def __init__(
        self,
        *,
        metric_specs: Sequence[MetricQuerySpec],
        project: str,
        experiment_name: str,
        standard_experiment_name: str | None = None,
        worker: str | None = None,
        replica: str | None = None,
        labels: Mapping[str, str] | None = None,
        standard_labels: Mapping[str, str] | None = None,
        inference_labels: Mapping[str, str] | None = None,
        association_targets: Sequence[str] | None = None,
        task_id: str | None = None,
        task_metadata: Mapping[str, Any] | None = None,
        config_dir: str | Path | None = None,
        query_step_seconds: float = DEFAULT_QUERY_STEP_SECONDS,
        baseline_window_seconds: float = DEFAULT_BASELINE_WINDOW_SECONDS,
        inference_window_seconds: float = DEFAULT_INFERENCE_WINDOW_SECONDS,
        endpoint_override: str | None = None,
        rl_insight_server_url: str | None = None,
        timeout_seconds: float = 30.0,
        bearer_token: str | None = None,
        use_environment_proxy: bool = False,
        endpoint_resolver: Callable[..., str] = resolve_prometheus_endpoint,
        query_client_factory: Callable[..., PrometheusQueryClient] = (
            PrometheusQueryClient
        ),
        detector_factory: Callable[..., DegradationPerception] = (
            DegradationPerception
        ),
    ) -> None:
        if not metric_specs:
            raise RLInsightIntegrationError(
                "metrics_required",
                "at least one metric is required",
            )
        self.metric_specs = list(metric_specs)
        self.project = _required_text(project, "project")
        self.experiment_name = _required_text(
            experiment_name,
            "experiment_name",
        )
        self.standard_experiment_name = (
            _required_text(
                standard_experiment_name,
                "standard_experiment_name",
            )
            if standard_experiment_name
            else self.experiment_name
        )
        self.worker = str(worker or "")
        self.replica = str(replica or "")
        self.labels = _string_mapping(labels)
        self.standard_labels = _string_mapping(standard_labels)
        self.inference_labels = _string_mapping(inference_labels)
        self.association_targets = (
            list(association_targets)
            if association_targets is not None
            else None
        )
        self.task_id = task_id or self.experiment_name
        self.task_metadata = dict(task_metadata or {})
        self.config_dir = (
            Path(config_dir).expanduser()
            if config_dir
            else Path.home() / ".rl-insight" / "degradation" / "config"
        )
        self.query_step_seconds = _positive_finite(
            query_step_seconds,
            "query_step_seconds",
        )
        self.baseline_window_seconds = _positive_finite(
            baseline_window_seconds,
            "baseline_window_seconds",
        )
        self.inference_window_seconds = _positive_finite(
            inference_window_seconds,
            "inference_window_seconds",
        )
        self.endpoint = endpoint_resolver(
            override=endpoint_override,
            rl_insight_server_url=rl_insight_server_url,
        )
        self.endpoint_source = (
            "development_override"
            if endpoint_override
            else "rl_insight_service_discovery"
        )
        self.client = query_client_factory(
            self.endpoint,
            timeout_seconds=timeout_seconds,
            bearer_token=bearer_token,
            use_environment_proxy=use_environment_proxy,
        )
        self.detector_factory = detector_factory
        self.detector: DegradationPerception | None = None
        self.last_windows: DetectionWindows | None = None
        self.last_diagnostics: dict[str, Any] = {}

    def run_once(
        self,
        *,
        now: float | None = None,
        standard_start: float | None = None,
        standard_end: float | None = None,
        inference_start: float | None = None,
        inference_end: float | None = None,
    ) -> dict[str, Any]:
        """Run one complete query, conversion, and detection cycle."""

        windows = derive_detection_windows(
            now=now,
            standard_start=standard_start,
            standard_end=standard_end,
            inference_start=inference_start,
            inference_end=inference_end,
            task_metadata=self.task_metadata,
            baseline_window_seconds=self.baseline_window_seconds,
            inference_window_seconds=self.inference_window_seconds,
        )
        self.last_windows = windows
        phase_labels = {
            "standard": self._phase_labels(
                self.standard_experiment_name,
                self.standard_labels,
            ),
            "inference": self._phase_labels(
                self.experiment_name,
                self.inference_labels,
            ),
        }
        dataset: dict[str, dict[str, Any]] = {
            "standard": {},
            "inference": {},
        }
        diagnostics: dict[str, Any] = {
            "endpointSource": self.endpoint_source,
            "windows": windows.to_dict(),
            "queries": {"standard": {}, "inference": {}},
        }
        for phase in ("standard", "inference"):
            window = getattr(windows, phase)
            for spec in self.metric_specs:
                query = spec.query_for(phase_labels[phase])
                response = self.client.query_range(
                    query,
                    start=window.start,
                    end=window.end,
                    step=self.query_step_seconds,
                )
                converted, adapter_diagnostics = convert_matrix_response(
                    response,
                    spec.logical_name,
                    phase,
                    query,
                    spec.series_policy,
                    spec.select_labels,
                    query_window=window.to_dict(),
                )
                dataset[phase][spec.logical_name] = converted
                diagnostics["queries"][phase][spec.logical_name] = {
                    "query": query,
                    "adapter": adapter_diagnostics,
                }
        metric_names = [spec.logical_name for spec in self.metric_specs]
        if self.detector is None:
            self.detector = self.detector_factory(
                dataset=dataset,
                metrics=metric_names,
                association_targets=self.association_targets,
                task_id=self.task_id,
                source_type="prometheus",
                config_dir=self.config_dir,
            )
            response = self.detector.detect()
        else:
            response = self.detector.detect_dataset(dataset, metrics=metric_names)
        self.last_diagnostics = diagnostics
        return response

    def metric_labels(self) -> dict[str, str]:
        return {
            "project": self.project,
            "experiment_name": self.experiment_name,
            "worker": self.worker,
            "replica": self.replica,
        }

    def _phase_labels(
        self,
        experiment_name: str,
        overrides: Mapping[str, str],
    ) -> dict[str, str]:
        selected = {
            **self.labels,
            "project": self.project,
            "experiment_name": experiment_name,
        }
        if self.worker:
            selected["worker"] = self.worker
        if self.replica:
            selected["replica"] = self.replica
        selected.update(overrides)
        return selected


class DerivedMetricPublisher:
    """Publish bounded degradation summaries through RL-Insight primitives."""

    def __init__(
        self,
        labels: Mapping[str, str],
        *,
        registry: MetricRegistry | None = None,
    ) -> None:
        self.labels = {
            "project": str(labels.get("project", "")),
            "experiment_name": str(labels.get("experiment_name", "")),
            "worker": str(labels.get("worker", "")),
            "replica": str(labels.get("replica", "")),
        }
        self.registry = registry or MetricRegistry(
            namespace="rl_insight",
            subsystem="degradation",
        )
        self._previous_associations: set[tuple[str, str, str]] = set()

    def start_endpoint(
        self,
        *,
        port: int,
        bind_address: str = "",
        advertise_host: str | None = None,
        server_starter: Callable[..., None] = start_metrics_http_server,
        target_registrar: Callable[..., None] = update_prometheus_config,
    ) -> str:
        """Expose `/metrics` and register it as a scrape target."""

        server_starter(int(port), addr=bind_address)
        host = str(advertise_host or local_addresses().get("host") or "").strip()
        if not host:
            raise RLInsightIntegrationError(
                "metrics_advertise_host_missing",
                "could not determine an address reachable by Prometheus",
            )
        target = format_host_port(host, int(port))
        target_registrar([target], job_name=DEGRADATION_SCRAPE_JOB)
        return f"http://{target}/metrics"

    def publish(
        self,
        response: Mapping[str, Any],
        *,
        detected_at: float | None = None,
    ) -> None:
        """Convert a complete DetectionResponse into bounded gauge series."""

        timestamp = float(time.time() if detected_at is None else detected_at)
        states = response.get("states", {})
        results = response.get("results", {})
        metric_errors = response.get("metricErrors", {})
        metric_names = list(
            dict.fromkeys(
                [
                    *list(states if isinstance(states, Mapping) else {}),
                    *list(results if isinstance(results, Mapping) else {}),
                    *list(metric_errors if isinstance(metric_errors, Mapping) else {}),
                ]
            )
        )
        self._gauge(
            "run_success",
            0.0 if metric_errors else 1.0,
            "Whether the latest degradation run completed without metric errors.",
            {},
        )
        for metric in metric_names:
            result = (
                results.get(metric, {})
                if isinstance(results, Mapping)
                else {}
            )
            state = (
                float(states[metric])
                if isinstance(states, Mapping) and metric in states
                else -1.0
            )
            published_ranges = _ranges_for(
                response.get("abnormalTimeRange", {}),
                metric,
            )
            current_ranges = _sequence_of_mappings(
                result.get("currentAbnormalTimeRange", [])
                if isinstance(result, Mapping)
                else []
            )
            diagnostics = _sequence_of_mappings(
                result.get("pointDiagnostics", [])
                if isinstance(result, Mapping)
                else []
            )
            abnormal_points = (
                sum(bool(item.get("abnormal")) for item in diagnostics)
                if diagnostics
                else sum(
                    int(item.get("abnormalPointCount", 0))
                    for item in current_ranges
                )
            )
            point_rate = (
                abnormal_points / len(diagnostics) if diagnostics else 0.0
            )
            interval_rate = max(
                (
                    float(item.get("abnormalRate", 0.0))
                    for item in current_ranges
                ),
                default=0.0,
            )
            thresholds = _sequence_of_mappings(
                result.get("thresholds", [])
                if isinstance(result, Mapping)
                else []
            )
            lower = min(
                (
                    float(item["lower_threshold"])
                    for item in thresholds
                    if item.get("lower_threshold") is not None
                ),
                default=float("nan"),
            )
            upper = max(
                (
                    float(item["upper_threshold"])
                    for item in thresholds
                    if item.get("upper_threshold") is not None
                ),
                default=float("nan"),
            )
            metric_label = {"metric": metric}
            self._gauge(
                "abnormal",
                float(bool(published_ranges)),
                "Whether a confirmed degradation interval exists.",
                metric_label,
            )
            self._gauge(
                "detection_state",
                state,
                "Latest degradation detector state; -1 is an execution error.",
                metric_label,
            )
            self._gauge(
                "lower_threshold",
                lower,
                "Lower envelope of the active KDE threshold models.",
                metric_label,
            )
            self._gauge(
                "upper_threshold",
                upper,
                "Upper envelope of the active KDE threshold models.",
                metric_label,
            )
            self._gauge(
                "abnormal_rate",
                max(point_rate, interval_rate),
                "Abnormal sample ratio in the latest inference window.",
                metric_label,
            )
            self._gauge(
                "abnormal_point_count",
                float(abnormal_points),
                "Abnormal sample count in the latest inference window.",
                metric_label,
            )
            self._gauge(
                "last_detected_timestamp_seconds",
                timestamp,
                "Unix timestamp of the latest completed detection.",
                metric_label,
            )
            self._gauge(
                "interval_active",
                float(bool(current_ranges)),
                "Whether the latest run contains an active abnormal interval.",
                metric_label,
            )
        self._publish_associations(response)

    def _publish_associations(self, response: Mapping[str, Any]) -> None:
        current: set[tuple[str, str, str]] = set()
        association = response.get("associationAnalysis", {})
        targets = (
            association.get("targets", {})
            if isinstance(association, Mapping)
            else {}
        )
        if isinstance(targets, Mapping):
            for target, target_result in targets.items():
                events = _sequence_of_mappings(
                    target_result.get("events", [])
                    if isinstance(target_result, Mapping)
                    else []
                )
                if not events:
                    continue
                top_items = _sequence_of_mappings(
                    events[-1].get("topAssociations", [])
                )
                for item in top_items:
                    key = (
                        str(target),
                        str(item.get("metric", "")),
                        str(item.get("rank", "")),
                    )
                    current.add(key)
                    self._association_gauge(
                        key,
                        float(item.get("abnormalContribution", 0.0)),
                    )
        for stale in self._previous_associations - current:
            self._association_gauge(stale, 0.0)
        self._previous_associations = current

    def _association_gauge(
        self,
        key: tuple[str, str, str],
        value: float,
    ) -> None:
        target, associated_metric, rank = key
        self._gauge(
            "association_score_percent",
            value,
            "Top-K degradation association score in percent.",
            {
                "target_metric": target,
                "associated_metric": associated_metric,
                "rank": rank,
            },
        )

    def _gauge(
        self,
        name: str,
        value: float,
        documentation: str,
        labels: Mapping[str, str],
    ) -> None:
        self.registry.value(
            name,
            documentation,
            float(value),
            self.labels,
            labels,
        )


def parse_timestamp(value: str | float | int | None) -> float | None:
    """Parse epoch seconds or an ISO-8601 timestamp for CLI/API callers."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise RLInsightIntegrationError(
            "invalid_timestamp",
            "timestamp must be epoch seconds or ISO-8601 text",
        )
    if isinstance(value, (int, float)):
        return _optional_timestamp(value, "timestamp")
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return _optional_timestamp(float(raw), "timestamp")
    except (TypeError, ValueError, RLInsightIntegrationError):
        pass
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise RLInsightIntegrationError(
            "invalid_timestamp",
            f"invalid timestamp: {raw!r}",
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _label_selector(labels: Mapping[str, str]) -> str:
    if not labels:
        return ""
    matchers: list[str] = []
    for name, value in sorted(labels.items()):
        if not _LABEL_NAME.fullmatch(str(name)):
            raise RLInsightIntegrationError(
                "invalid_label_name",
                f"invalid Prometheus label name: {name!r}",
            )
        escaped = (
            str(value)
            .replace("\\", "\\\\")
            .replace("\n", "\\n")
            .replace('"', '\\"')
        )
        matchers.append(f'{name}="{escaped}"')
    return "{" + ",".join(matchers) + "}"


def _ranges_for(value: Any, metric: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    return _sequence_of_mappings(value.get(metric, []))


def _sequence_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _positive_finite(value: Any, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise RLInsightIntegrationError(
            "invalid_duration",
            f"{name} must be a positive finite number",
        ) from exc
    if not math.isfinite(converted) or converted <= 0:
        raise RLInsightIntegrationError(
            "invalid_duration",
            f"{name} must be a positive finite number",
        )
    return converted


def _optional_timestamp(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise RLInsightIntegrationError(
            "invalid_timestamp",
            f"{name} must be finite epoch seconds",
        ) from exc
    if not math.isfinite(converted):
        raise RLInsightIntegrationError(
            "invalid_timestamp",
            f"{name} must be finite epoch seconds",
        )
    return converted


def _required_text(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise RLInsightIntegrationError(
            "required_metadata_missing",
            f"{name} is required",
        )
    return result


def _string_mapping(value: Mapping[str, Any] | None) -> dict[str, str]:
    return {
        str(key): str(item)
        for key, item in dict(value or {}).items()
    }


__all__ = [
    "DEFAULT_BASELINE_WINDOW_SECONDS",
    "DEFAULT_INFERENCE_WINDOW_SECONDS",
    "DEFAULT_QUERY_STEP_SECONDS",
    "DerivedMetricPublisher",
    "DetectionWindow",
    "DetectionWindows",
    "MetricQuerySpec",
    "PrometheusDegradationRunner",
    "PrometheusQueryError",
    "RLInsightIntegrationError",
    "derive_detection_windows",
    "parse_timestamp",
    "resolve_prometheus_endpoint",
]
