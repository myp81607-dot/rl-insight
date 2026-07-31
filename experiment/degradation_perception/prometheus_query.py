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

"""Prometheus HTTP read client used by degradation integrations."""

from __future__ import annotations

import copy
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any


MAX_RESPONSE_BYTES = 64 * 1024 * 1024


class PrometheusQueryError(ValueError):
    """Structured Prometheus endpoint, transport, or response failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = str(code)
        self.message = str(message)
        self.details = copy.deepcopy(dict(details or {}))
        super().__init__(f"{self.code}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": copy.deepcopy(self.details),
        }


def validate_prometheus_endpoint(value: str) -> str:
    """Return a normalized HTTP(S) endpoint without credentials or query data."""

    endpoint = str(value).strip().rstrip("/")
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PrometheusQueryError(
            "invalid_prometheus_endpoint",
            "Prometheus endpoint must be a valid HTTP(S) URL",
        )
    if parsed.username is not None or parsed.password is not None:
        raise PrometheusQueryError(
            "invalid_prometheus_endpoint",
            "Prometheus endpoint must not contain embedded credentials",
        )
    if parsed.query or parsed.fragment:
        raise PrometheusQueryError(
            "invalid_prometheus_endpoint",
            "Prometheus endpoint must not contain a query or fragment",
        )
    return endpoint


class PrometheusQueryClient:
    """Small strict client for Prometheus instant and range query APIs."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_seconds: float = 30.0,
        bearer_token: str | None = None,
        use_environment_proxy: bool = False,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
    ) -> None:
        self.endpoint = validate_prometheus_endpoint(endpoint)
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise PrometheusQueryError(
                "invalid_prometheus_timeout",
                "Prometheus timeout must be positive",
            )
        self.bearer_token = bearer_token
        self.use_environment_proxy = bool(use_environment_proxy)
        self.max_response_bytes = int(max_response_bytes)
        if self.max_response_bytes <= 0:
            raise PrometheusQueryError(
                "invalid_prometheus_response_limit",
                "Prometheus response byte limit must be positive",
            )

    def query(
        self,
        query: str,
        *,
        time: str | float | None = None,
    ) -> dict[str, Any]:
        """Issue one ``/api/v1/query`` request."""

        parameters: dict[str, Any] = {"query": str(query)}
        if time is not None:
            parameters["time"] = time
        return self._get("/api/v1/query", parameters)

    def query_range(
        self,
        query: str,
        *,
        start: str | float,
        end: str | float,
        step: float,
    ) -> dict[str, Any]:
        """Issue one ``/api/v1/query_range`` request."""

        return self._get(
            "/api/v1/query_range",
            {
                "query": str(query),
                "start": start,
                "end": end,
                "step": f"{float(step):g}",
            },
        )

    def _get(self, path: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
        encoded = urllib.parse.urlencode(dict(parameters))
        url = f"{self.endpoint}{path}?{encoded}"
        headers = {
            "Accept": "application/json",
            "User-Agent": "rl-insight-degradation-perception/1",
        }
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            if self.use_environment_proxy:
                opened = urllib.request.urlopen(
                    request,
                    timeout=self.timeout_seconds,
                )
            else:
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({})
                )
                opened = opener.open(request, timeout=self.timeout_seconds)
            with opened as response:
                body = response.read(self.max_response_bytes + 1)
        except urllib.error.HTTPError as exc:
            try:
                body_excerpt = exc.read(4096).decode(
                    "utf-8",
                    errors="replace",
                )
            except OSError:
                body_excerpt = ""
            raise PrometheusQueryError(
                "prometheus_http_error",
                f"Prometheus returned HTTP {exc.code}",
                details={
                    "status": exc.code,
                    "bodyExcerpt": body_excerpt,
                },
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PrometheusQueryError(
                "prometheus_connection_error",
                f"failed to contact Prometheus: {exc}",
            ) from exc
        if len(body) > self.max_response_bytes:
            raise PrometheusQueryError(
                "prometheus_response_too_large",
                "Prometheus response exceeds the configured byte limit",
                details={"maximumBytes": self.max_response_bytes},
            )
        try:
            decoded = body.decode("utf-8")
            payload = json.loads(
                decoded,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise PrometheusQueryError(
                "invalid_prometheus_json",
                f"Prometheus response is not strict UTF-8 JSON: {exc}",
            ) from exc
        if not isinstance(payload, Mapping):
            raise PrometheusQueryError(
                "invalid_prometheus_json",
                "Prometheus response root must be an object",
            )
        return dict(payload)


def fetch_query_range(
    *,
    base_url: str,
    query: str,
    start: str | float,
    end: str | float,
    step: float,
    timeout_seconds: float,
    bearer_token: str | None = None,
    use_environment_proxy: bool = False,
) -> dict[str, Any]:
    """Compatibility function for the legacy workflow entry point."""

    return PrometheusQueryClient(
        base_url,
        timeout_seconds=timeout_seconds,
        bearer_token=bearer_token,
        use_environment_proxy=use_environment_proxy,
    ).query_range(query, start=start, end=end, step=step)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")
