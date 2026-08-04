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

"""One-shot degradation detection for local VeRL training logs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baseline_model import (
    BaselineDetectionRunner,
    fit_baseline_model,
    load_baseline_model,
    save_baseline_model,
)
from .algorithm_policy import (
    DEFAULT_ALGORITHM_POLICY_PATH,
    load_algorithm_policy_config,
)
from .detection_runtime import (
    DetectionRunner,
    format_terminal_summary,
    save_result,
)
from .perception_config import DEFAULT_METRIC, SUPPORTED_SOURCE_TYPES
from .policy import resolve_metric_policy
from .training_log import load_verl_training_log
from .two_step_analysis import (
    analyze_detection_event,
    attach_abnormal_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiment.degradation_perception.main",
        description=(
            "Learn a fixed KDE baseline from a healthy VeRL log and detect "
            "degradation in a separate inference log."
        ),
    )
    parser.add_argument(
        "--standard-log",
        type=Path,
        required=True,
        help="UTF-8 healthy VeRL step-metrics log used only as standard data.",
    )
    parser.add_argument(
        "--inference-log",
        type=Path,
        help="UTF-8 VeRL step-metrics log used only as inference data.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[DEFAULT_METRIC],
        help="Original metric names to detect; '/', '_' and '.' are preserved.",
    )
    parser.add_argument(
        "--policy-config",
        type=Path,
        default=DEFAULT_ALGORITHM_POLICY_PATH,
        help="YAML file containing algorithm and history parameters.",
    )
    parser.add_argument(
        "--source-type",
        choices=sorted(SUPPORTED_SOURCE_TYPES),
        default="training_log",
        help="Timestamp display convention passed to the existing algorithm.",
    )
    parser.add_argument(
        "--task-id",
        default="degradation-log-run",
        help="Task identifier stored in the JSON result.",
    )
    parser.add_argument(
        "--association-targets",
        nargs="+",
        default=None,
        help="Optional explicit targets for the existing association analysis.",
    )
    parser.add_argument(
        "--debug-kde",
        action="store_true",
        help="Print alpha, quantile probabilities, and every KDE model range.",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="Print each parsed original metric name once, then exit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination JSON file (required unless --list-metrics is used).",
    )
    return parser


def run_offline_detection(args: argparse.Namespace) -> dict[str, Any]:
    """Parse two independent files and call the shared detection runtime."""

    standard = load_verl_training_log(args.standard_log)
    inference = load_verl_training_log(args.inference_log)
    policy = load_algorithm_policy_config(
        getattr(args, "policy_config", DEFAULT_ALGORITHM_POLICY_PATH)
    )
    runner = DetectionRunner(
        standard=standard,
        metrics=args.metrics,
        task_id=args.task_id,
        source_type=args.source_type,
        association_targets=args.association_targets,
        metric_policies=policy.for_metrics(args.metrics),
        history_policy=policy.history_policy,
    )
    return runner.run(inference)


def _legacy_main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        standard = load_verl_training_log(args.standard_log)
        if args.list_metrics:
            for metric in standard:
                sys.stdout.write(metric + "\n")
            return 0
        if args.inference_log is None:
            parser.error("--inference-log is required unless --list-metrics is used")
        if args.output is None:
            parser.error("--output is required unless --list-metrics is used")

        # Use the already parsed standard mapping so the healthy file is read
        # exactly once and can never be confused with inference data.
        inference = load_verl_training_log(args.inference_log)
        policy = load_algorithm_policy_config(args.policy_config)
        runner = DetectionRunner(
            standard=standard,
            metrics=args.metrics,
            task_id=args.task_id,
            source_type=args.source_type,
            association_targets=args.association_targets,
            metric_policies=policy.for_metrics(args.metrics),
            history_policy=policy.history_policy,
        )
        result = runner.run(inference)
        destination = save_result(result, args.output)
        sys.stdout.write(
            format_terminal_summary(result, debug_kde=args.debug_kde)
        )
        sys.stdout.write(f"Result JSON: {destination}\n")
        return 0
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1


@dataclass(frozen=True)
class RealTestConfig:
    baseline_model: Path
    inference_log: Path
    metrics: tuple[str, ...]
    task_id: str
    debug_kde: bool
    output: Path
    policy_config: Path | None = None


def build_fit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m experiment.degradation_perception.main fit',
        description='Fit and atomically save a normal KDE baseline.',
    )
    parser.add_argument('--standard-log', type=Path, required=True)
    parser.add_argument('--metrics', nargs='+', required=True)
    parser.add_argument('--output-model', type=Path, required=True)
    parser.add_argument('--baseline-id', default='default')
    parser.add_argument(
        '--policy-config',
        type=Path,
        default=DEFAULT_ALGORITHM_POLICY_PATH,
    )
    parser.add_argument(
        '--source-type',
        choices=sorted(SUPPORTED_SOURCE_TYPES),
        default='training_log',
    )
    return parser


def build_detect_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m experiment.degradation_perception.main detect',
        description='Load a saved baseline and detect one local VeRL log.',
    )
    parser.add_argument('--config', type=Path, required=True)
    return parser


def build_analyze_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m experiment.degradation_perception.main analyze',
        description='Analyze one previously detected abnormal event.',
    )
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--inference-log', type=Path, required=True)
    parser.add_argument('--target-metric', required=True)
    parser.add_argument('--event-index', type=int, required=True)
    parser.add_argument('--top-k', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--policy-config',
        type=Path,
        default=DEFAULT_ALGORITHM_POLICY_PATH,
    )
    return parser


def _invalid_json_number(value: str) -> None:
    raise ValueError(f'config contains invalid JSON number {value}')


def _config_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value.strip()


def _config_relative_path(base: Path, value: Any, name: str) -> Path:
    raw = Path(_config_text(value, name)).expanduser()
    return raw.resolve() if raw.is_absolute() else (base / raw).resolve()


def load_detection_result(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(
            source.read_text(encoding='utf-8'),
            parse_constant=_invalid_json_number,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f'could not load detection result: {exc}') from exc
    if not isinstance(raw, dict):
        raise ValueError('detection result root must be an object')
    return raw


def load_real_test_config(path: str | Path) -> RealTestConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(
            config_path.read_text(encoding='utf-8'),
            parse_constant=_invalid_json_number,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'could not load detect config: {exc}') from exc
    if not isinstance(raw, dict):
        raise ValueError('detect config root must be an object')
    required = {
        'baseline_model',
        'inference_log',
        'metrics',
        'task_id',
        'debug_kde',
        'output',
    }
    missing = required - set(raw)
    unknown = set(raw) - (required | {'policy_config'})
    if missing:
        raise ValueError(f'detect config is missing fields: {sorted(missing)}')
    if unknown:
        raise ValueError(f'detect config has unsupported fields: {sorted(unknown)}')

    raw_metrics = raw['metrics']
    if (
        not isinstance(raw_metrics, list)
        or not raw_metrics
        or any(not isinstance(metric, str) or not metric.strip() for metric in raw_metrics)
    ):
        raise ValueError('metrics must be a non-empty list of metric names')
    metrics = tuple(metric.strip() for metric in raw_metrics)
    if len(set(metrics)) != len(metrics):
        raise ValueError('metrics must not contain duplicates')
    if not isinstance(raw['debug_kde'], bool):
        raise ValueError('debug_kde must be a JSON boolean')

    base = config_path.parent
    baseline_path = _config_relative_path(
        base, raw['baseline_model'], 'baseline_model'
    )
    inference_path = _config_relative_path(
        base, raw['inference_log'], 'inference_log'
    )
    output_path = _config_relative_path(base, raw['output'], 'output')
    policy_path = (
        None
        if raw.get('policy_config') in (None, '')
        else _config_relative_path(base, raw['policy_config'], 'policy_config')
    )
    for input_path, name in (
        (baseline_path, 'baseline_model'),
        (inference_path, 'inference_log'),
    ):
        if not input_path.is_file():
            raise ValueError(f'{name} is not an existing file: {input_path}')
    if policy_path is not None and not policy_path.is_file():
        raise ValueError(f'policy_config is not an existing file: {policy_path}')
    protected_paths = {config_path, baseline_path, inference_path}
    if policy_path is not None:
        protected_paths.add(policy_path)
    if output_path in protected_paths:
        raise ValueError('output must not overwrite the config or either input file')
    return RealTestConfig(
        baseline_model=baseline_path,
        inference_log=inference_path,
        metrics=metrics,
        task_id=_config_text(raw['task_id'], 'task_id'),
        debug_kde=raw['debug_kde'],
        output=output_path,
        policy_config=policy_path,
    )


def _fit_main(argv: Sequence[str]) -> int:
    args = build_fit_parser().parse_args(list(argv))
    try:
        standard = load_verl_training_log(args.standard_log)
        policy = load_algorithm_policy_config(args.policy_config)
        model = fit_baseline_model(
            standard,
            args.metrics,
            baseline_id=args.baseline_id,
            source_type=args.source_type,
            metric_policies=policy.for_metrics(args.metrics),
            history_policy=policy.history_policy,
        )
        destination = save_baseline_model(model, args.output_model)
        sys.stdout.write(f'Baseline model: {destination}\n')
        sys.stdout.write(f'Baseline ID: {model.baseline_id}\n')
        for metric, baseline in model.metrics.items():
            sys.stdout.write(
                f'{metric}: {len(baseline.models)} normal model(s), '
                f'{baseline.standard_samples} standard samples\n'
            )
        return 0
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        sys.stderr.write(f'error: {exc}\n')
        return 1


def _detect_main(argv: Sequence[str]) -> int:
    args = build_detect_parser().parse_args(list(argv))
    try:
        config = load_real_test_config(args.config)
        baseline = load_baseline_model(
            config.baseline_model,
            expected_metrics=config.metrics,
        )
        inference = load_verl_training_log(config.inference_log)
        policy = (
            None
            if config.policy_config is None
            else load_algorithm_policy_config(config.policy_config)
        )
        runner = BaselineDetectionRunner(
            baseline,
            metrics=config.metrics,
            task_id=config.task_id,
            metric_policies=(
                None if policy is None else policy.for_metrics(config.metrics)
            ),
            history_policy=(None if policy is None else policy.history_policy),
        )
        result = attach_abnormal_metrics(runner.run(inference))
        destination = save_result(result, config.output)
        sys.stdout.write(
            format_terminal_summary(result, debug_kde=config.debug_kde)
        )
        sys.stdout.write(f'Result JSON: {destination}\n')
        return 0
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        sys.stderr.write(f'error: {exc}\n')
        return 1


def _analyze_main(argv: Sequence[str]) -> int:
    args = build_analyze_parser().parse_args(list(argv))
    try:
        result_path = args.result.expanduser().resolve()
        inference_path = args.inference_log.expanduser().resolve()
        output_path = args.output.expanduser().resolve()
        if output_path == result_path:
            raise ValueError('output must not overwrite the detection result')
        if output_path == inference_path:
            raise ValueError('output must not overwrite the inference log')

        detection_result = load_detection_result(result_path)
        inference = load_verl_training_log(inference_path)
        policy = load_algorithm_policy_config(args.policy_config)
        metric_policy = resolve_metric_policy(
            args.target_metric,
            policy.metric_policy,
        )
        analysis = analyze_detection_event(
            detection_result,
            inference,
            target_metric=args.target_metric,
            event_index=args.event_index,
            top_k=args.top_k,
            association_config=metric_policy['association'],
        )
        destination = save_result(analysis, output_path)
        sys.stdout.write(
            f"Analyzed {args.target_metric} event {args.event_index}; "
            f"returned {len(analysis['topAssociations'])} association(s).\n"
        )
        sys.stdout.write(f'Association JSON: {destination}\n')
        return 0
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        sys.stderr.write(f'error: {exc}\n')
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == 'fit':
        return _fit_main(raw[1:])
    if raw and raw[0] == 'detect':
        return _detect_main(raw[1:])
    if raw and raw[0] == 'analyze':
        return _analyze_main(raw[1:])
    return _legacy_main(raw)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run_offline_detection"]
