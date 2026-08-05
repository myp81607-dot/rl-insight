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

"""Small Prometheus HTTP adapter for degradation-perception experiments."""

from __future__ import annotations

import json
import math
import os
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .perception_config import TimeSeries


class PrometheusSourceError(RuntimeError):
    """Base error raised by the Prometheus adapter."""


class PrometheusRequestError(PrometheusSourceError):
    """The HTTP request or Prometheus API response failed."""


class PrometheusDataError(PrometheusSourceError):
    """Prometheus returned data that cannot be mapped unambiguously."""


def _promql_string(value: str) -> str:
    """Quote one PromQL label value without accepting raw PromQL fragments."""

    return json.dumps(value, ensure_ascii=False)


@dataclass(frozen=True)
class PrometheusSeries:
    """One concrete Prometheus series returned by ``/api/v1/series``."""

    metric_name: str
    labels: dict[str, str]

    @classmethod
    def from_label_set(cls, raw: Mapping[str, Any]) -> "PrometheusSeries":
        if not isinstance(raw, Mapping):
            raise PrometheusDataError("Prometheus series entry must be an object")
        metric_name = raw.get("__name__")
        if not isinstance(metric_name, str) or not metric_name:
            raise PrometheusDataError(
                "Prometheus series entry is missing a non-empty __name__"
            )
        labels: dict[str, str] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise PrometheusDataError(
                    "Prometheus series labels must be string pairs"
                )
            if key != "__name__":
                labels[key] = value
        return cls(metric_name=metric_name, labels=labels)

    @property
    def selector(self) -> str:
        matchers = [f'__name__={_promql_string(self.metric_name)}']
        matchers.extend(
            f'{key}={_promql_string(value)}'
            for key, value in sorted(self.labels.items())
        )
        return "{" + ",".join(matchers) + "}"

    @property
    def identifier(self) -> str:
        if not self.labels:
            return self.metric_name
        labels = ",".join(
            f'{key}={_promql_string(value)}'
            for key, value in sorted(self.labels.items())
        )
        return f"{self.metric_name}{{{labels}}}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "metricName": self.metric_name,
            "labels": dict(sorted(self.labels.items())),
            "selector": self.selector,
        }


@dataclass(frozen=True)
class BootstrapWindow:
    """Step-indexed healthy samples and their source wall-clock boundaries."""

    series: dict[str, TimeSeries]
    observed_steps: tuple[int, ...]
    start_timestamp: float
    end_timestamp: float


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _parse_matrix_result(payload: Any, *, query: str) -> TimeSeries:
    if not isinstance(payload, Mapping):
        raise PrometheusDataError("Prometheus response must be a JSON object")
    if payload.get("status") != "success":
        error_type = payload.get("errorType", "api_error")
        message = payload.get("error", "Prometheus query failed")
        raise PrometheusRequestError(f"{error_type}: {message}")
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise PrometheusDataError("Prometheus response data must be an object")
    if data.get("resultType") != "matrix":
        raise PrometheusDataError(
            "query_range must return resultType 'matrix', got "
            f"{data.get('resultType')!r}"
        )
    result = data.get("result")
    if not isinstance(result, list):
        raise PrometheusDataError("Prometheus matrix result must be an array")
    if not result:
        return TimeSeries()
    if len(result) != 1:
        label_sets = [
            item.get("metric", {}) if isinstance(item, Mapping) else {}
            for item in result[:5]
        ]
        raise PrometheusDataError(
            f"PromQL for {query!r} returned {len(result)} series; "
            "add label filters or an explicit PromQL aggregation. "
            f"First label sets: {label_sets!r}"
        )
    entry = result[0]
    if not isinstance(entry, Mapping):
        raise PrometheusDataError("Prometheus matrix entry must be an object")
    values = entry.get("values")
    if not isinstance(values, list):
        raise PrometheusDataError("Prometheus matrix entry must contain a values array")

    # Prometheus values are [unix_timestamp, "sample"] pairs.  Keep the last
    # finite value for a duplicate timestamp, matching the training-log parser.
    points: dict[float, float] = {}
    for item in values:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise PrometheusDataError(
                "Prometheus samples must be [timestamp, value] pairs"
            )
        try:
            timestamp = float(item[0])
            value = float(item[1])
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(timestamp) and math.isfinite(value):
            points[timestamp] = value
    ordered = sorted(points.items())
    return TimeSeries(
        timestamps=[timestamp for timestamp, _ in ordered],
        values=[value for _, value in ordered],
    )


class PrometheusHttpClient:
    """Read range vectors from an existing Prometheus HTTP API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 30.0,
        verify_tls: bool = True,
        bearer_token_env: str | None = None,
        username_env: str | None = None,
        password_env: str | None = None,
        session: Any | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("Prometheus base_url must be a non-empty string")
        self.base_url = base_url.strip().rstrip("/")
        self.timeout_seconds = _positive_float(
            timeout_seconds,
            "timeout_seconds",
        )
        if not isinstance(verify_tls, bool):
            raise TypeError("verify_tls must be a boolean")
        self.verify_tls = verify_tls

        if session is None:
            try:
                import requests
            except ImportError as exc:  # pragma: no cover - packaging guard.
                raise PrometheusRequestError(
                    "requests is required for Prometheus monitoring"
                ) from exc
            session = requests.Session()
        self.session = session
        self.headers: dict[str, str] = {"Accept": "application/json"}
        if bearer_token_env is not None:
            token = os.environ.get(bearer_token_env)
            if not token:
                raise ValueError(
                    f"environment variable {bearer_token_env!r} is not set"
                )
            self.headers["Authorization"] = f"Bearer {token}"

        if (username_env is None) != (password_env is None):
            raise ValueError(
                "username_env and password_env must be configured together"
            )
        self.auth: tuple[str, str] | None = None
        if username_env is not None and password_env is not None:
            username = os.environ.get(username_env)
            password = os.environ.get(password_env)
            if username is None or password is None:
                raise ValueError(
                    "configured Prometheus basic-auth environment variables "
                    "are not both set"
                )
            self.auth = (username, password)

    def list_series(
        self,
        match: str = '{__name__=~".+"}',
        *,
        start: float,
        end: float,
    ) -> list[PrometheusSeries]:
        """Return every concrete label set present in the requested interval."""

        if not isinstance(match, str) or not match.strip():
            raise ValueError("series match selector must be a non-empty string")
        start_number = _finite_float(start, "start")
        end_number = _finite_float(end, "end")
        if start_number > end_number:
            raise ValueError("start must not exceed end")
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/series",
                params={
                    "match[]": match.strip(),
                    "start": start_number,
                    "end": end_number,
                },
                headers=dict(self.headers),
                auth=self.auth,
                timeout=self.timeout_seconds,
                verify=self.verify_tls,
            )
            response.raise_for_status()
            payload = response.json()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            raise PrometheusRequestError(
                f"Prometheus series discovery failed for {match!r}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise PrometheusDataError("Prometheus response must be a JSON object")
        if payload.get("status") != "success":
            error_type = payload.get("errorType", "api_error")
            message = payload.get("error", "Prometheus series discovery failed")
            raise PrometheusRequestError(f"{error_type}: {message}")
        raw_series = payload.get("data")
        if not isinstance(raw_series, list):
            raise PrometheusDataError("Prometheus series data must be an array")

        unique: dict[str, PrometheusSeries] = {}
        for raw in raw_series:
            series = PrometheusSeries.from_label_set(raw)
            unique[series.identifier] = series
        return [unique[key] for key in sorted(unique)]

    def query_range(
        self,
        query: str,
        *,
        start: float,
        end: float,
        step: float,
    ) -> TimeSeries:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("PromQL query must be a non-empty string")
        start_number = _finite_float(start, "start")
        end_number = _finite_float(end, "end")
        step_number = _positive_float(step, "step")
        if start_number > end_number:
            raise ValueError("start must not exceed end")
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/query_range",
                params={
                    "query": query.strip(),
                    "start": start_number,
                    "end": end_number,
                    "step": step_number,
                },
                headers=dict(self.headers),
                auth=self.auth,
                timeout=self.timeout_seconds,
                verify=self.verify_tls,
            )
            response.raise_for_status()
            payload = response.json()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            raise PrometheusRequestError(
                f"Prometheus query_range failed for {query!r}: {exc}"
            ) from exc
        return _parse_matrix_result(payload, query=query)

def _finite_points(series: TimeSeries) -> list[tuple[float, float]]:
    points: dict[float, float] = {}
    for timestamp, value in zip(series.timestamps, series.values):
        try:
            parsed_timestamp = float(timestamp)
            parsed_value = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(parsed_timestamp) and math.isfinite(parsed_value):
            points[parsed_timestamp] = parsed_value
    return sorted(points.items())


def _nearest_point(
    points: Sequence[tuple[float, float]],
    timestamp: float,
    tolerance: float,
) -> tuple[float, float] | None:
    if not points:
        return None
    timestamps = [point[0] for point in points]
    insertion = bisect_left(timestamps, timestamp)
    candidates: list[tuple[float, float]] = []
    if insertion < len(points):
        candidates.append(points[insertion])
    if insertion > 0:
        candidates.append(points[insertion - 1])
    nearest = min(candidates, key=lambda item: (abs(item[0] - timestamp), item[0]))
    return nearest if abs(nearest[0] - timestamp) <= tolerance else None


def _integer_step(value: float) -> int | None:
    rounded = round(value)
    if abs(value - rounded) > 1.0e-6:
        return None
    return int(rounded)


def build_bootstrap_window(
    global_steps: TimeSeries,
    metrics: Mapping[str, TimeSeries],
    *,
    start_step: int,
    end_step: int,
    alignment_tolerance_seconds: float,
) -> BootstrapWindow:
    """Select at most one aligned sample per global step for baseline fitting."""

    if isinstance(start_step, bool) or not isinstance(start_step, int):
        raise TypeError("start_step must be an integer")
    if isinstance(end_step, bool) or not isinstance(end_step, int):
        raise TypeError("end_step must be an integer")
    if start_step > end_step:
        raise ValueError("start_step must not exceed end_step")
    tolerance = _positive_float(
        alignment_tolerance_seconds,
        "alignment_tolerance_seconds",
    )

    # A range query can repeat a gauge value.  Retain the latest wall-clock
    # timestamp for each distinct integer training step.
    step_timestamps: dict[int, float] = {}
    for timestamp, raw_step in _finite_points(global_steps):
        step = _integer_step(raw_step)
        if step is not None and start_step <= step <= end_step:
            step_timestamps[step] = timestamp
    if not step_timestamps:
        raise PrometheusDataError(
            f"global-step query contains no steps in [{start_step}, {end_step}]"
        )
    ordered_steps = sorted(step_timestamps)
    aligned: dict[str, TimeSeries] = {}
    for metric, series in metrics.items():
        points = _finite_points(series)
        timestamps: list[float] = []
        values: list[float] = []
        for step in ordered_steps:
            match = _nearest_point(points, step_timestamps[step], tolerance)
            if match is None:
                continue
            timestamps.append(float(step))
            values.append(match[1])
        aligned[metric] = TimeSeries(timestamps=timestamps, values=values)
    source_timestamps = [step_timestamps[step] for step in ordered_steps]
    return BootstrapWindow(
        series=aligned,
        observed_steps=tuple(ordered_steps),
        start_timestamp=min(source_timestamps),
        end_timestamp=max(source_timestamps),
    )


def filter_after_global_step(
    global_steps: TimeSeries,
    metrics: Mapping[str, TimeSeries],
    *,
    minimum_step_exclusive: int,
    alignment_tolerance_seconds: float,
) -> dict[str, TimeSeries]:
    """Keep wall-clock metric samples whose nearest global step is newer."""

    if isinstance(minimum_step_exclusive, bool) or not isinstance(
        minimum_step_exclusive, int
    ):
        raise TypeError("minimum_step_exclusive must be an integer")
    tolerance = _positive_float(
        alignment_tolerance_seconds,
        "alignment_tolerance_seconds",
    )
    step_points = _finite_points(global_steps)
    filtered: dict[str, TimeSeries] = {}
    for metric, series in metrics.items():
        timestamps: list[float] = []
        values: list[float] = []
        for timestamp, value in _finite_points(series):
            match = _nearest_point(step_points, timestamp, tolerance)
            if match is None:
                continue
            step = _integer_step(match[1])
            if step is None or step <= minimum_step_exclusive:
                continue
            timestamps.append(timestamp)
            values.append(value)
        filtered[metric] = TimeSeries(timestamps=timestamps, values=values)
    return filtered


def latest_integer_step(global_steps: TimeSeries) -> int | None:
    """Return the latest finite integer global-step value, if present."""

    points = _finite_points(global_steps)
    for _timestamp, value in reversed(points):
        step = _integer_step(value)
        if step is not None:
            return step
    return None


__all__ = [
    "BootstrapWindow",
    "PrometheusDataError",
    "PrometheusHttpClient",
    "PrometheusSeries",
    "PrometheusRequestError",
    "PrometheusSourceError",
    "build_bootstrap_window",
    "filter_after_global_step",
    "latest_integer_step",
]
