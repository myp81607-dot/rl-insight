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
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .detection_runtime import (
    DetectionRunner,
    format_terminal_summary,
    save_result,
)
from .perception_config import DEFAULT_METRIC, SUPPORTED_SOURCE_TYPES
from .training_log import load_verl_training_log


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
    runner = DetectionRunner(
        standard=standard,
        metrics=args.metrics,
        task_id=args.task_id,
        source_type=args.source_type,
        association_targets=args.association_targets,
    )
    return runner.run(inference)


def main(argv: Sequence[str] | None = None) -> int:
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
        runner = DetectionRunner(
            standard=standard,
            metrics=args.metrics,
            task_id=args.task_id,
            source_type=args.source_type,
            association_targets=args.association_targets,
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


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run_offline_detection"]
