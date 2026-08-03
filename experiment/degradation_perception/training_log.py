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

"""Parse metric time series from exact VeRL ``step:<N>`` log records."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .perception_config import TimeSeries
from .timeseries import preprocess_time_series


@dataclass(frozen=True)
class TrainingLogParseResult:
    """Parsed series plus source-order valid steps for polling callers."""

    series: dict[str, TimeSeries]
    valid_steps: list[int]


def parse_verl_training_log_line(
    line: str,
    *,
    min_step_exclusive: int | None = None,
) -> tuple[int, dict[str, float]] | None:
    """Parse one exact ``step:<N> - metric:value`` record.

    Invalid fields are skipped independently. A line is valid only when its
    first segment is named exactly ``step`` and at least one finite numeric
    metric remains.
    """

    if not isinstance(line, str):
        raise TypeError("training log lines must be strings")
    parts = line.strip().split(" - ")
    if not parts or ":" not in parts[0]:
        return None
    step_key, step_text = parts[0].split(":", 1)
    if step_key.strip() != "step":
        return None
    try:
        step = int(step_text.strip())
    except (TypeError, ValueError, OverflowError):
        return None
    if min_step_exclusive is not None and step <= min_step_exclusive:
        return None

    metrics: dict[str, float] = {}
    for field in parts[1:]:
        if ":" not in field:
            continue
        metric_name, value_text = field.split(":", 1)
        metric_name = metric_name.strip()
        if not metric_name:
            continue
        try:
            value = float(value_text.strip())
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value):
            metrics[metric_name] = value
    return (step, metrics) if metrics else None


def parse_verl_training_log_with_metadata(
    text_or_lines: str | Iterable[str],
    *,
    min_step_exclusive: int | None = None,
) -> TrainingLogParseResult:
    """Parse records and retain valid source steps for remote polling."""

    lines = (
        text_or_lines.splitlines()
        if isinstance(text_or_lines, str)
        else text_or_lines
    )
    samples: dict[str, tuple[list[int], list[float]]] = {}
    valid_steps: list[int] = []
    for line in lines:
        parsed = parse_verl_training_log_line(
            line,
            min_step_exclusive=min_step_exclusive,
        )
        if parsed is None:
            continue
        step, metrics = parsed
        valid_steps.append(step)
        for metric_name, value in metrics.items():
            timestamps, values = samples.setdefault(metric_name, ([], []))
            timestamps.append(step)
            values.append(value)

    return TrainingLogParseResult(
        series={
            metric_name: preprocess_time_series(timestamps, values)
            for metric_name, (timestamps, values) in samples.items()
        },
        valid_steps=valid_steps,
    )


def parse_verl_training_log(
    text_or_lines: str | Iterable[str],
    *,
    min_step_exclusive: int | None = None,
) -> dict[str, TimeSeries]:
    """Return first-seen metric series parsed from exact VeRL step lines."""

    return parse_verl_training_log_with_metadata(
        text_or_lines,
        min_step_exclusive=min_step_exclusive,
    ).series


def load_verl_training_log(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    min_step_exclusive: int | None = None,
) -> dict[str, TimeSeries]:
    """Parse metric time series from a local VeRL log file."""

    with Path(path).open("r", encoding=encoding) as stream:
        return parse_verl_training_log(
            stream,
            min_step_exclusive=min_step_exclusive,
        )


__all__ = [
    "TrainingLogParseResult",
    "load_verl_training_log",
    "parse_verl_training_log",
    "parse_verl_training_log_line",
    "parse_verl_training_log_with_metadata",
]
