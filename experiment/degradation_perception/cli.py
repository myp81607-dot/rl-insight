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

"""CLI registration and standalone entry point for degradation integration."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .prometheus_matrix_adapter import PrometheusMatrixError
from .prometheus_query import PrometheusQueryError
from .rl_insight_integration import (
    DEFAULT_BASELINE_WINDOW_SECONDS,
    DEFAULT_INFERENCE_WINDOW_SECONDS,
    DEFAULT_QUERY_STEP_SECONDS,
    DerivedMetricPublisher,
    MetricQuerySpec,
    PrometheusDegradationRunner,
    RLInsightIntegrationError,
    parse_timestamp,
)


def add_degradation_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register ``rl-insight degradation`` on a root parser."""

    degradation = subparsers.add_parser(
        "degradation",
        help="Analyze and continuously monitor performance degradation.",
    )
    _add_actions(degradation)
    return degradation


def _build_standalone_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rl-insight-degradation",
        description=(
            "RL-Insight Prometheus degradation analysis without a workflow YAML."
        ),
    )
    _add_actions(parser)
    return parser


def _add_actions(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(
        dest="degradation_command",
        required=True,
    )
    analyze = actions.add_parser(
        "analyze",
        help="Run one detection and print the complete DetectionResponse JSON.",
    )
    _add_analysis_arguments(analyze)
    analyze.set_defaults(func=_run_analyze)

    monitor = actions.add_parser(
        "monitor",
        help="Run detections continuously and publish derived metrics.",
    )
    _add_analysis_arguments(monitor)
    monitor.add_argument(
        "--interval-seconds",
        type=float,
        default=60.0,
        help="Delay between completed detection runs.",
    )
    monitor.add_argument(
        "--metrics-port",
        type=int,
        default=9093,
        help="Port for the degradation /metrics endpoint.",
    )
    monitor.add_argument(
        "--metrics-bind-address",
        default="",
        help="Bind address for the degradation /metrics endpoint.",
    )
    monitor.add_argument(
        "--metrics-advertise-host",
        default=None,
        help="Host reachable by managed Prometheus; normally auto-detected.",
    )
    monitor.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    monitor.set_defaults(func=_run_monitor)


def _add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument(
        "--standard-experiment-name",
        default=None,
        help=(
            "Healthy baseline experiment; defaults to --experiment-name "
            "with an earlier time window."
        ),
    )
    parser.add_argument(
        "--metric",
        action="append",
        required=True,
        metavar="LOGICAL[=PROMETHEUS_METRIC]",
        help="Repeat for every metric consumed by the detector.",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        metavar="LOGICAL=PROMQL_TEMPLATE",
        help=(
            "Optional advanced query. Use {{metric}} and {{labels}} placeholders "
            "to reuse RL-Insight task labels."
        ),
    )
    parser.add_argument(
        "--association-target",
        action="append",
        default=None,
        help="Repeat to enable existing Top-K association analysis.",
    )
    parser.add_argument("--worker", default=None)
    parser.add_argument("--replica", default=None)
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional label applied to both windows.",
    )
    parser.add_argument(
        "--standard-label",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument(
        "--inference-label",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument("--standard-start", default=None)
    parser.add_argument("--standard-end", default=None)
    parser.add_argument("--inference-start", default=None)
    parser.add_argument("--inference-end", default=None)
    parser.add_argument(
        "--task-start",
        default=None,
        help="Optional task metadata start time used before default windows.",
    )
    parser.add_argument(
        "--baseline-window-seconds",
        type=float,
        default=DEFAULT_BASELINE_WINDOW_SECONDS,
    )
    parser.add_argument(
        "--inference-window-seconds",
        type=float,
        default=DEFAULT_INFERENCE_WINDOW_SECONDS,
    )
    parser.add_argument(
        "--query-step-seconds",
        type=float,
        default=DEFAULT_QUERY_STEP_SECONDS,
    )
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for DetectionResponse and query diagnostics.",
    )
    parser.add_argument(
        "--prometheus-url",
        default=None,
        help="Development/test override; production uses RL-Insight discovery.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--bearer-token-env",
        default=None,
        help="Read an optional Prometheus bearer token from this environment key.",
    )
    parser.add_argument(
        "--use-environment-proxy",
        action="store_true",
    )


def _run_analyze(args: argparse.Namespace) -> int:
    runner = _runner_from_args(args)
    response = _run_once_from_args(runner, args)
    _write_outputs(args.output_dir, response, runner.last_diagnostics)
    _print_json(response)
    return 0


def _run_monitor(args: argparse.Namespace) -> int:
    if args.interval_seconds <= 0:
        raise RLInsightIntegrationError(
            "invalid_monitor_interval",
            "--interval-seconds must be positive",
        )
    if args.iterations is not None and args.iterations <= 0:
        raise RLInsightIntegrationError(
            "invalid_monitor_iterations",
            "--iterations must be positive",
        )
    runner = _runner_from_args(args)
    publisher = DerivedMetricPublisher(runner.metric_labels())
    publisher.start_endpoint(
        port=args.metrics_port,
        bind_address=args.metrics_bind_address,
        advertise_host=args.metrics_advertise_host,
    )
    completed = 0
    while args.iterations is None or completed < args.iterations:
        detected_at = time.time()
        response = _run_once_from_args(
            runner,
            args,
            now=detected_at,
        )
        publisher.publish(response, detected_at=detected_at)
        _write_outputs(args.output_dir, response, runner.last_diagnostics)
        _print_json(response)
        completed += 1
        if args.iterations is None or completed < args.iterations:
            time.sleep(float(args.interval_seconds))
    return 0


def _runner_from_args(args: argparse.Namespace) -> PrometheusDegradationRunner:
    queries = _assignments(args.query, "--query")
    specs: list[MetricQuerySpec] = []
    for item in args.metric:
        logical_name, prometheus_metric = _metric_assignment(item)
        specs.append(
            MetricQuerySpec(
                logical_name=logical_name,
                prometheus_metric=prometheus_metric,
                query_template=queries.get(logical_name),
            )
        )
    unknown_queries = sorted(set(queries) - {spec.logical_name for spec in specs})
    if unknown_queries:
        raise RLInsightIntegrationError(
            "query_metric_unknown",
            "--query references metrics not declared by --metric",
            details={"metrics": unknown_queries},
        )
    token = None
    if args.bearer_token_env:
        token = os.environ.get(args.bearer_token_env)
        if not token:
            raise RLInsightIntegrationError(
                "bearer_token_missing",
                f"environment variable {args.bearer_token_env!r} is not set",
            )
    task_start = parse_timestamp(args.task_start)
    return PrometheusDegradationRunner(
        metric_specs=specs,
        project=args.project,
        experiment_name=args.experiment_name,
        standard_experiment_name=args.standard_experiment_name,
        worker=args.worker,
        replica=args.replica,
        labels=_assignments(args.label, "--label"),
        standard_labels=_assignments(
            args.standard_label,
            "--standard-label",
        ),
        inference_labels=_assignments(
            args.inference_label,
            "--inference-label",
        ),
        association_targets=args.association_target,
        task_id=args.task_id,
        task_metadata=(
            {"start_time": task_start} if task_start is not None else None
        ),
        config_dir=args.config_dir,
        query_step_seconds=args.query_step_seconds,
        baseline_window_seconds=args.baseline_window_seconds,
        inference_window_seconds=args.inference_window_seconds,
        endpoint_override=args.prometheus_url,
        timeout_seconds=args.timeout_seconds,
        bearer_token=token,
        use_environment_proxy=args.use_environment_proxy,
    )


def _run_once_from_args(
    runner: PrometheusDegradationRunner,
    args: argparse.Namespace,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    explicit_inference_end = parse_timestamp(args.inference_end)
    return runner.run_once(
        now=now,
        standard_start=parse_timestamp(args.standard_start),
        standard_end=parse_timestamp(args.standard_end),
        inference_start=parse_timestamp(args.inference_start),
        inference_end=explicit_inference_end,
    )


def _metric_assignment(value: str) -> tuple[str, str]:
    raw = str(value).strip()
    if "=" not in raw:
        if not raw:
            raise RLInsightIntegrationError(
                "invalid_metric_argument",
                "--metric must not be empty",
            )
        return raw, raw
    logical_name, prometheus_metric = raw.split("=", 1)
    if not logical_name.strip() or not prometheus_metric.strip():
        raise RLInsightIntegrationError(
            "invalid_metric_argument",
            "--metric must be LOGICAL or LOGICAL=PROMETHEUS_METRIC",
        )
    return logical_name.strip(), prometheus_metric.strip()


def _assignments(values: Sequence[str], argument: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        raw = str(value)
        if "=" not in raw:
            raise RLInsightIntegrationError(
                "invalid_assignment",
                f"{argument} requires KEY=VALUE",
            )
        key, item = raw.split("=", 1)
        key = key.strip()
        item = item.strip()
        if not key or not item:
            raise RLInsightIntegrationError(
                "invalid_assignment",
                f"{argument} requires non-empty KEY=VALUE",
            )
        if key in result:
            raise RLInsightIntegrationError(
                "duplicate_assignment",
                f"{argument} repeats key {key!r}",
            )
        result[key] = item
    return result


def _write_outputs(
    output_dir: Path | None,
    response: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> None:
    if output_dir is None:
        return
    output = output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "detection_response.json", response)
    _write_json(output / "query_diagnostics.json", diagnostics)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(serialized + "\n", encoding="utf-8")


def _print_json(value: Mapping[str, Any]) -> None:
    sys.stdout.write(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def _error_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(
        exc,
        (
            RLInsightIntegrationError,
            PrometheusQueryError,
            PrometheusMatrixError,
        ),
    ):
        error = exc.to_dict()
    else:
        error = {
            "code": type(exc).__name__,
            "message": str(exc),
            "details": {},
        }
    return {"ok": False, "error": error}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the installed standalone command and emit strict JSON on failure."""

    args = _build_standalone_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        _print_json(_error_payload(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["add_degradation_parser", "main"]
