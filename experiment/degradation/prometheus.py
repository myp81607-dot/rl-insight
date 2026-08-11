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

"""Read degradation experiment time series from the Prometheus HTTP API."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

import requests

from .metrics import GLOBAL_STEP_METRIC

DEFAULT_GLOBAL_STEP_METRIC = GLOBAL_STEP_METRIC
DEFAULT_SERIES_SELECTOR = '{__name__=~".+"}'


class PrometheusError(RuntimeError):
    """Base error raised by the Prometheus client."""


class PrometheusRequestError(PrometheusError):
    """The HTTP request or Prometheus API operation failed."""


class PrometheusResponseError(PrometheusError):
    """Prometheus returned an unexpected response."""


@dataclass(frozen=True, order=True)
class SeriesId:
    """A Prometheus series identified by its metric name and labels."""

    name: str
    labels: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_label_set(cls, value: Mapping[str, Any]) -> SeriesId:
        """Build a stable identifier from a Prometheus metric label set."""

        if not isinstance(value, Mapping):
            raise PrometheusResponseError("series labels must be an object")
        name = value.get("__name__")
        if not isinstance(name, str) or not name:
            raise PrometheusResponseError("series is missing a metric name")

        labels: list[tuple[str, str]] = []
        for key, label_value in value.items():
            if not isinstance(key, str) or not isinstance(label_value, str):
                raise PrometheusResponseError("series labels must be strings")
            if key != "__name__":
                labels.append((key, label_value))
        return cls(name=name, labels=tuple(sorted(labels)))


@dataclass(frozen=True)
class Sample:
    """One finite Prometheus sample."""

    timestamp: float
    value: float


@dataclass(frozen=True)
class GlobalStep:
    """The latest completed training step and its Prometheus timestamp."""

    timestamp: float
    value: int


@dataclass(frozen=True)
class TimeSeries:
    """Samples for one concrete Prometheus series."""

    identity: SeriesId
    samples: tuple[Sample, ...]


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        requirement = "a positive number" if positive else "a finite number"
        raise ValueError(f"{name} must be {requirement}")
    return number


def _sample(value: Any) -> Sample | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise PrometheusResponseError(
            "Prometheus samples must be [timestamp, value] pairs"
        )
    try:
        timestamp = float(value[0])
        sample_value = float(value[1])
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(timestamp) or not math.isfinite(sample_value):
        return None
    return Sample(timestamp=timestamp, value=sample_value)


class PrometheusClient:
    """Small, stateless adapter around the Prometheus query API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 30.0,
        verify_tls: bool = True,
        session: requests.Session | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(verify_tls, bool):
            raise TypeError("verify_tls must be a boolean")

        self._base_url = base_url.strip().rstrip("/")
        self._timeout = _number(timeout_seconds, "timeout_seconds", positive=True)
        self._verify_tls = verify_tls
        self._owns_session = session is None
        self._session = session or requests.Session()

    def close(self) -> None:
        """Close a session created by this client."""

        if self._owns_session:
            self._session.close()

    def _get(self, path: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            response = self._session.get(
                f"{self._base_url}{path}",
                params=dict(params),
                timeout=self._timeout,
                verify=self._verify_tls,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise PrometheusRequestError(f"Prometheus request failed: {exc}") from exc

        if not isinstance(payload, Mapping):
            raise PrometheusResponseError("Prometheus response must be an object")
        if payload.get("status") != "success":
            error_type = payload.get("errorType", "api_error")
            message = payload.get("error", "Prometheus query failed")
            raise PrometheusRequestError(f"{error_type}: {message}")
        return payload

    @staticmethod
    def _query_result(payload: Mapping[str, Any], result_type: str) -> list[Any]:
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise PrometheusResponseError("Prometheus data must be an object")
        if data.get("resultType") != result_type:
            raise PrometheusResponseError(
                f"expected resultType {result_type!r}, got {data.get('resultType')!r}"
            )
        result = data.get("result")
        if not isinstance(result, list):
            raise PrometheusResponseError("Prometheus result must be an array")
        return result

    def query_global_step(
        self,
        metric_name: str = DEFAULT_GLOBAL_STEP_METRIC,
        *,
        time: float | None = None,
    ) -> GlobalStep:
        """Return the single current integer-valued global-step series."""

        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValueError("metric_name must be a non-empty string")
        params: dict[str, Any] = {"query": metric_name.strip()}
        if time is not None:
            params["time"] = _number(time, "time")

        result = self._query_result(
            self._get("/api/v1/query", params), result_type="vector"
        )
        if len(result) != 1:
            raise PrometheusResponseError(
                f"{metric_name!r} must return exactly one series, got {len(result)}"
            )
        entry = result[0]
        if not isinstance(entry, Mapping):
            raise PrometheusResponseError("global-step result must be an object")
        metric = entry.get("metric")
        if not isinstance(metric, Mapping):
            raise PrometheusResponseError("global-step result is missing labels")
        identity = SeriesId.from_label_set(metric)
        if identity.name != metric_name.strip():
            raise PrometheusResponseError(
                f"expected metric {metric_name!r}, got {identity.name!r}"
            )

        sample = _sample(entry.get("value"))
        if sample is None:
            raise PrometheusResponseError("global-step sample must be finite")
        step = round(sample.value)
        if not math.isclose(sample.value, step, abs_tol=1e-9):
            raise PrometheusResponseError("global-step value must be an integer")
        return GlobalStep(timestamp=sample.timestamp, value=step)

    def discover_series(
        self,
        *,
        start: float,
        end: float,
        selector: str = DEFAULT_SERIES_SELECTOR,
    ) -> tuple[SeriesId, ...]:
        """Return series present in a Prometheus time interval."""

        start_value, end_value = self._range(start, end)
        if not isinstance(selector, str) or not selector.strip():
            raise ValueError("selector must be a non-empty string")
        payload = self._get(
            "/api/v1/series",
            {
                "match[]": selector.strip(),
                "start": start_value,
                "end": end_value,
            },
        )
        result = payload.get("data")
        if not isinstance(result, list):
            raise PrometheusResponseError("Prometheus series data must be an array")
        return tuple(sorted({SeriesId.from_label_set(item) for item in result}))

    def query_range(
        self,
        *,
        start: float,
        end: float,
        step: float,
        query: str = DEFAULT_SERIES_SELECTOR,
        series: Collection[SeriesId] | None = None,
    ) -> tuple[TimeSeries, ...]:
        """Read multiple time series with one Prometheus range query."""

        start_value, end_value = self._range(start, end)
        step_value = _number(step, "step", positive=True)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        selected = frozenset(series) if series is not None else None
        if selected is not None and not selected:
            return ()
        result = self._query_result(
            self._get(
                "/api/v1/query_range",
                {
                    "query": query.strip(),
                    "start": start_value,
                    "end": end_value,
                    "step": step_value,
                },
            ),
            result_type="matrix",
        )

        points: dict[SeriesId, dict[float, float]] = (
            {identity: {} for identity in selected} if selected is not None else {}
        )
        for item in result:
            if not isinstance(item, Mapping):
                raise PrometheusResponseError("matrix entry must be an object")
            metric = item.get("metric")
            values = item.get("values")
            if not isinstance(metric, Mapping) or not isinstance(values, list):
                raise PrometheusResponseError(
                    "matrix entry must contain metric and values"
                )
            identity = SeriesId.from_label_set(metric)
            if selected is not None and identity not in selected:
                continue
            series_points = points.setdefault(identity, {})
            for raw_sample in values:
                sample = _sample(raw_sample)
                if sample is not None:
                    series_points[sample.timestamp] = sample.value

        return tuple(
            TimeSeries(
                identity=identity,
                samples=tuple(
                    Sample(timestamp=timestamp, value=value)
                    for timestamp, value in sorted(series_points.items())
                ),
            )
            for identity, series_points in sorted(points.items())
        )

    @staticmethod
    def _range(start: float, end: float) -> tuple[float, float]:
        start_value = _number(start, "start")
        end_value = _number(end, "end")
        if start_value > end_value:
            raise ValueError("start must not exceed end")
        return start_value, end_value


__all__ = [
    "DEFAULT_GLOBAL_STEP_METRIC",
    "DEFAULT_SERIES_SELECTOR",
    "GlobalStep",
    "PrometheusClient",
    "PrometheusError",
    "PrometheusRequestError",
    "PrometheusResponseError",
    "Sample",
    "SeriesId",
    "TimeSeries",
]
