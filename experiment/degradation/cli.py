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

"""Command-line entry point for the degradation experiment."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .baseline import BaselineParameters
from .config import (
    DEFAULT_ABNORMAL_DATA_PATH,
    DEFAULT_PROMETHEUS_URL,
    DEFAULT_STANDARD_DATA_PATH,
    DEFAULT_TARGET_METRICS,
    DegradationConfig,
    RuntimeParameters,
)
from .prometheus import (
    DEFAULT_GLOBAL_STEP_METRIC,
    DEFAULT_SERIES_SELECTOR,
    PrometheusClient,
    PrometheusError,
)
from .runtime import DegradationRuntime, DegradationRuntimeError
from .storage import StorageError

LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    defaults = RuntimeParameters()
    baseline = BaselineParameters()
    parser = argparse.ArgumentParser(
        prog="python -m experiment.degradation.cli",
        description="Train or load a degradation baseline and monitor new steps.",
    )
    parser.add_argument("--prometheus-url", default=DEFAULT_PROMETHEUS_URL)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for standard_data.json and abnormal_data.json.",
    )
    parser.add_argument(
        "--target-metric",
        action="append",
        default=None,
        help="Additional UP-policy target metric; repeat to add more.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=defaults.poll_interval_seconds,
        help="Prometheus polling interval in seconds (default: 300).",
    )
    parser.add_argument(
        "--query-step",
        type=float,
        default=defaults.query_step_seconds,
        help="Prometheus range-query resolution in seconds (default: 10).",
    )
    parser.add_argument(
        "--baseline-steps",
        type=int,
        default=None,
        help="Number of complete steps used to fit a new baseline (default: 40).",
    )
    parser.add_argument(
        "--pre-context-steps",
        type=int,
        default=defaults.pre_context_steps,
        help="Steps retained before an event for association (default: 30).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=defaults.top_k,
        help="Associated metrics saved per event phase (default: 5).",
    )
    parser.add_argument(
        "--baseline-ratio",
        "--ratio",
        dest="baseline_ratio",
        type=float,
        default=None,
        help="Set both fitted baseline expansion ratios (default: 1.05).",
    )
    parser.add_argument(
        "--baseline-alpha",
        type=float,
        default=None,
        help=f"KDE tail probability for a new baseline (default: {baseline.alpha}).",
    )
    parser.add_argument(
        "--minimum-samples",
        type=int,
        default=None,
        help=(
            "Minimum valid values required per baseline series "
            f"(default: {baseline.minimum_samples})."
        ),
    )
    parser.add_argument(
        "--global-step-metric",
        default=DEFAULT_GLOBAL_STEP_METRIC,
    )
    parser.add_argument(
        "--series-selector",
        default=DEFAULT_SERIES_SELECTOR,
        help=(
            "PromQL selector for series discovery and range queries; the "
            "global-step instant query must still be unique in v0.1."
        ),
    )
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for Prometheus requests.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete both state files before initialization and retrain baseline.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Initialize, perform one detection poll, and exit.",
    )
    parser.add_argument(
        "--show-components",
        action="store_true",
        help="Print stored correlation and random-forest event details, then exit.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def _config_from_args(
    args: argparse.Namespace,
) -> tuple[DegradationConfig, bool]:
    defaults = RuntimeParameters()
    targets = tuple(
        dict.fromkeys((*DEFAULT_TARGET_METRICS, *(args.target_metric or ())))
    )
    fit_overridden = any(
        value is not None
        for value in (
            args.baseline_steps,
            args.baseline_ratio,
            args.baseline_alpha,
            args.minimum_samples,
        )
    )
    runtime = RuntimeParameters(
        baseline_step_count=(
            defaults.baseline_step_count
            if args.baseline_steps is None
            else args.baseline_steps
        ),
        poll_interval_seconds=args.poll_interval,
        query_step_seconds=args.query_step,
        pre_context_steps=args.pre_context_steps,
        top_k=args.top_k,
        target_metrics=targets,
        candidate_metrics=tuple(
            metric for metric in defaults.candidate_metrics if metric not in targets
        ),
        global_step_metric=args.global_step_metric,
        series_selector=args.series_selector,
    )
    baseline = BaselineParameters()
    if args.baseline_ratio is not None:
        baseline = replace(
            baseline,
            lower_ratio=args.baseline_ratio,
            upper_ratio=args.baseline_ratio,
        )
    if args.baseline_alpha is not None:
        baseline = replace(baseline, alpha=args.baseline_alpha)
    if args.minimum_samples is not None:
        baseline = replace(baseline, minimum_samples=args.minimum_samples)

    state_dir = args.state_dir
    return (
        DegradationConfig(
            prometheus_url=args.prometheus_url,
            standard_data_path=state_dir / DEFAULT_STANDARD_DATA_PATH.name,
            abnormal_data_path=state_dir / DEFAULT_ABNORMAL_DATA_PATH.name,
            request_timeout_seconds=args.request_timeout,
            verify_tls=not args.insecure,
            runtime=runtime,
            baseline=baseline,
        ),
        fit_overridden,
    )


def _show_components(path: Path) -> int:
    if not path.exists():
        LOGGER.error("No abnormal data found at %s", path)
        return 1
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.error("Cannot read %s: %s", path, exc)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone degradation experiment CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    try:
        config, fit_overridden = _config_from_args(args)
        if args.show_components:
            return _show_components(config.abnormal_data_path)

        client = PrometheusClient(
            config.prometheus_url,
            timeout_seconds=config.request_timeout_seconds,
            verify_tls=config.verify_tls,
        )
        try:
            runtime = DegradationRuntime(client, config)
            if args.reset:
                runtime.reset()
                LOGGER.info("Cleared baseline and abnormal event history")
            runtime.initialize(require_baseline_match=fit_overridden and not args.reset)
            if args.once:
                runtime.run_once()
            else:
                runtime.run_forever()
        finally:
            client.close()
    except (
        DegradationRuntimeError,
        PrometheusError,
        StorageError,
        TypeError,
        ValueError,
    ) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
