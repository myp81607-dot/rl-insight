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
import importlib.util
import json
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import CANDIDATE_METRICS, TARGET_METRICS
from .presentation import PresentationError, present_state
from .state_schema import StateStatus
from .storage import (
    ABNORMAL_DATA_NAME,
    STANDARD_DATA_NAME,
    JsonStorage,
    StorageError,
)

LOGGER = logging.getLogger(__name__)
SUBCOMMANDS = frozenset({"check", "run-once", "monitor", "show", "reset"})
DEFAULT_PROMETHEUS_URL = "http://127.0.0.1:9090"
DEFAULT_GLOBAL_STEP_METRIC = "rl_insight_monitor_training_global_step"
DEFAULT_SERIES_SELECTOR = '{__name__=~".+"}'


class CommandError(RuntimeError):
    """A structured subcommand cannot complete its requested operation."""


def _print_json(value: object, *, stream: Any | None = None) -> None:
    print(
        json.dumps(value, indent=2, sort_keys=True),
        file=sys.stdout if stream is None else stream,
    )


def _add_runtime_arguments(
    parser: argparse.ArgumentParser, *, legacy_controls: bool
) -> None:
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
        default=None,
        help="Prometheus polling interval in seconds (default: 300).",
    )
    parser.add_argument(
        "--query-step",
        type=float,
        default=None,
        help="Prometheus range-query resolution in seconds (default: 10).",
    )
    parser.add_argument(
        "--baseline-steps",
        type=int,
        default=None,
        help="Number of complete steps used to fit a new baseline (default: 30).",
    )
    parser.add_argument(
        "--pre-context-steps",
        type=int,
        default=None,
        help="Steps retained before an event for association (default: 30).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Associated metrics saved per event phase (default: 25).",
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
        help="KDE tail probability for a new baseline (default: 0.025).",
    )
    parser.add_argument(
        "--minimum-samples",
        type=int,
        default=None,
        help="Minimum valid values required per baseline series (default: 20).",
    )
    parser.add_argument("--global-step-metric", default=DEFAULT_GLOBAL_STEP_METRIC)
    parser.add_argument(
        "--series-selector",
        default=DEFAULT_SERIES_SELECTOR,
        help=(
            "PromQL selector for discovery and range queries; the global-step "
            "instant query must still be unique."
        ),
    )
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for Prometheus requests.",
    )
    if legacy_controls:
        parser.add_argument(
            "--reset",
            action="store_true",
            help="Delete both state files, retrain the baseline, and continue.",
        )
        parser.add_argument(
            "--once",
            action="store_true",
            help="Initialize, perform one detection poll, and exit.",
        )
        parser.add_argument(
            "--show-components",
            action="store_true",
            help="Print the raw abnormal event JSON, then exit.",
        )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )


def _build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiment.degradation.cli",
        description="Train or load a degradation baseline and monitor new steps.",
    )
    _add_runtime_arguments(parser, legacy_controls=True)
    return parser


def _build_subcommand_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiment.degradation.cli",
        description="Operate and inspect the degradation experiment.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Run a read-only readiness check.")
    check.add_argument("--prometheus-url", default=DEFAULT_PROMETHEUS_URL)
    check.add_argument("--series-selector", default=DEFAULT_SERIES_SELECTOR)
    check.add_argument("--global-step-metric", default=DEFAULT_GLOBAL_STEP_METRIC)
    check.add_argument("--lookback-seconds", type=float, default=3600.0)
    check.add_argument("--request-timeout", type=float, default=30.0)
    check.add_argument("--state-dir", type=Path, default=Path.cwd())
    check.add_argument("--insecure", action="store_true")

    run_once = subparsers.add_parser(
        "run-once", help="Initialize and perform one detection poll."
    )
    _add_runtime_arguments(run_once, legacy_controls=False)

    monitor = subparsers.add_parser(
        "monitor", help="Run foreground degradation polling."
    )
    _add_runtime_arguments(monitor, legacy_controls=False)

    show = subparsers.add_parser("show", help="Print validated degradation state.")
    show.add_argument("--kind", choices=("all", "baseline", "events"), default="all")
    show.add_argument("--event", choices=("latest", "all"), default="latest")
    show.add_argument(
        "--phase",
        choices=("auto", "confirmed", "closed", "both"),
        default="auto",
    )
    show.add_argument("--state-dir", type=Path, default=Path.cwd())

    reset = subparsers.add_parser("reset", help="Preview or reset state safely.")
    reset.add_argument("--state-dir", type=Path, default=Path.cwd())
    reset.add_argument("--backup-dir", type=Path)
    reset.add_argument("--yes", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> tuple[Any, bool]:
    from .baseline import BaselineParameters
    from .config import DegradationConfig, RuntimeParameters

    defaults = RuntimeParameters()
    targets = tuple(dict.fromkeys((*TARGET_METRICS, *(args.target_metric or ()))))
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
        poll_interval_seconds=(
            defaults.poll_interval_seconds
            if args.poll_interval is None
            else args.poll_interval
        ),
        query_step_seconds=(
            defaults.query_step_seconds if args.query_step is None else args.query_step
        ),
        pre_context_steps=(
            defaults.pre_context_steps
            if args.pre_context_steps is None
            else args.pre_context_steps
        ),
        top_k=defaults.top_k if args.top_k is None else args.top_k,
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
            standard_data_path=state_dir / STANDARD_DATA_NAME,
            abnormal_data_path=state_dir / ABNORMAL_DATA_NAME,
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
    _print_json(payload)
    return 0


def _run_runtime(args: argparse.Namespace, *, once: bool, reset: bool) -> int:
    from .prometheus import PrometheusClient, PrometheusError
    from .runtime import DegradationRuntime, DegradationRuntimeError

    logging.basicConfig(level=getattr(logging, args.log_level))
    try:
        config, fit_overridden = _config_from_args(args)
        client = PrometheusClient(
            config.prometheus_url,
            timeout_seconds=config.request_timeout_seconds,
            verify_tls=config.verify_tls,
        )
        try:
            runtime = DegradationRuntime(client, config)
            if reset:
                runtime.reset()
                LOGGER.info("Cleared baseline and abnormal event history")
            runtime.initialize(require_baseline_match=fit_overridden and not reset)
            if once:
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


def _storage_from_state_dir(state_dir: Path) -> tuple[Path, JsonStorage]:
    directory = state_dir.expanduser().absolute()
    return directory, JsonStorage(
        directory / STANDARD_DATA_NAME,
        directory / ABNORMAL_DATA_NAME,
    )


def _dependency_status() -> dict[str, bool]:
    return {
        package: importlib.util.find_spec(module) is not None
        for package, module in (
            ("numpy", "numpy"),
            ("scipy", "scipy"),
            ("scikit-learn", "sklearn"),
        )
    }


def _check(args: argparse.Namespace) -> int:
    from .prometheus import PrometheusClient, PrometheusError

    try:
        client = PrometheusClient(
            args.prometheus_url,
            timeout_seconds=args.request_timeout,
            verify_tls=not args.insecure,
        )
        try:
            global_step = client.query_global_step(args.global_step_metric)
            end = max(time.time(), global_step.timestamp)
            series = client.discover_series(
                start=end - args.lookback_seconds,
                end=end,
                selector=args.series_selector,
            )
        finally:
            client.close()
    except (PrometheusError, OSError, TypeError, ValueError) as exc:
        raise CommandError(str(exc)) from exc

    metric_names = {identity.name for identity in series}
    targets = sorted(metric_names.intersection(TARGET_METRICS))
    candidates = sorted(metric_names.intersection(CANDIDATE_METRICS))
    _, storage = _storage_from_state_dir(args.state_dir)
    state = storage.inspect_state_files()
    dependencies = _dependency_status()
    warnings = []
    if not targets:
        warnings.append("no configured target metric discovered")
    if not candidates:
        warnings.append("no configured candidate metric discovered")
    missing_dependencies = sorted(
        package for package, available in dependencies.items() if not available
    )
    if missing_dependencies:
        warnings.append(
            "missing optional dependencies: " + ", ".join(missing_dependencies)
        )
    invalid_state = [
        item["path"]
        for item in state
        if item["exists"] and not item.get("valid_json", False)
    ]
    if invalid_state:
        warnings.append("invalid state file(s): " + ", ".join(invalid_state))
    inconsistent_state = state[1]["exists"] and not state[0]["exists"]
    if inconsistent_state:
        warnings.append(f"{ABNORMAL_DATA_NAME} exists without {STANDARD_DATA_NAME}")
    payload = {
        "ok": (
            bool(targets)
            and bool(candidates)
            and not missing_dependencies
            and not invalid_state
            and not inconsistent_state
        ),
        "prometheus_url": args.prometheus_url,
        "series_selector": args.series_selector,
        "global_step": {
            "metric": args.global_step_metric,
            "value": global_step.value,
            "timestamp": global_step.timestamp,
        },
        "discovery": {
            "lookback_seconds": args.lookback_seconds,
            "series_count": len(series),
            "metric_name_count": len(metric_names),
            "target_metric_count": len(targets),
            "target_metrics": targets,
            "candidate_metric_count": len(candidates),
        },
        "dependencies": dependencies,
        "state": state,
        "warnings": warnings,
    }
    if not candidates:
        payload["error_kind"] = "no_candidate_metrics"
    _print_json(payload)
    return 0 if payload["ok"] else 1


def _show(args: argparse.Namespace) -> int:
    _, storage = _storage_from_state_dir(args.state_dir)
    inspection = storage.inspect_state()
    payload = present_state(
        inspection,
        kind=args.kind,
        event=args.event,
        phase=args.phase,
    )
    _print_json(payload)
    return 1 if inspection.status is StateStatus.INVALID_STATE else 0


def _reset(args: argparse.Namespace) -> int:
    state_dir, storage = _storage_from_state_dir(args.state_dir)
    targets = storage.validate_state_targets()
    payload: dict[str, Any] = {
        "confirmed": args.yes,
        "state_dir": str(state_dir),
        "targets": [str(path) for path in targets],
    }
    if not args.yes or not targets:
        payload["action"] = "preview" if not args.yes else "nothing_to_reset"
        _print_json(payload)
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = (
        args.backup_dir.expanduser().resolve()
        if args.backup_dir is not None
        else state_dir / ".degradation-backups"
    )
    backup_dir = backup_root / stamp
    backups = storage.backup_state(backup_dir, targets)
    storage.delete_state(targets)
    payload.update(
        action="reset",
        backup_dir=str(backup_dir),
        backups=[str(path) for path in backups],
    )
    _print_json(payload)
    return 0


def _legacy_main(argv: Sequence[str]) -> int:
    parser = _build_legacy_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    try:
        config, _ = _config_from_args(args)
    except (TypeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    if args.show_components:
        return _show_components(config.abnormal_data_path)
    return _run_runtime(args, once=args.once, reset=args.reset)


def _subcommand_main(argv: Sequence[str]) -> int:
    parser = _build_subcommand_parser()
    args = parser.parse_args(argv)
    if args.command == "run-once":
        return _run_runtime(args, once=True, reset=False)
    if args.command == "monitor":
        return _run_runtime(args, once=False, reset=False)
    try:
        if args.command == "check":
            return _check(args)
        if args.command == "show":
            return _show(args)
        if args.command == "reset":
            return _reset(args)
        raise AssertionError(f"unhandled command: {args.command}")
    except (CommandError, PresentationError, StorageError, OSError, ValueError) as exc:
        _print_json({"ok": False, "error": str(exc)}, stream=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def main(argv: Sequence[str] | None = None) -> int:
    """Run either the subcommand interface or the compatible legacy CLI."""

    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] in SUBCOMMANDS:
        if len(values) > 1 and values[1] == "--":
            values.pop(1)
        return _subcommand_main(values)
    return _legacy_main(values)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
