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

"""Align Prometheus time series to training-step intervals."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .series import SeriesId, TimeSeries


class WindowError(ValueError):
    """The input time series cannot form valid step frames."""


class MissingGlobalStepError(WindowError):
    """A required global-step boundary is missing."""


@dataclass(frozen=True)
class StepFrame:
    """Metric snapshots for one half-open training-step interval."""

    step: int
    start_time: float
    end_time: float
    values: Mapping[SeriesId, float | None]


def _integer_step(value: float) -> int:
    if not math.isfinite(value):
        raise WindowError("global-step values must be finite integers")
    step = round(value)
    if not math.isclose(value, step, rel_tol=0.0, abs_tol=1e-9):
        raise WindowError("global-step values must be finite integers")
    return step


def _step_boundaries(global_steps: TimeSeries) -> dict[int, float]:
    samples = sorted(global_steps.samples, key=lambda sample: sample.timestamp)
    if not samples:
        raise MissingGlobalStepError("global-step series is empty")

    boundaries: dict[int, float] = {}
    previous: int | None = None
    for sample in samples:
        if not math.isfinite(sample.timestamp):
            raise WindowError("global-step timestamps must be finite")
        step = _integer_step(sample.value)
        if previous is not None and step < previous:
            raise WindowError("global-step values must be monotonic")
        boundaries.setdefault(step, sample.timestamp)
        previous = step
    return boundaries


def _finite_points(series: TimeSeries) -> tuple[list[float], list[float]]:
    points = sorted(
        (
            (sample.timestamp, sample.value)
            for sample in series.samples
            if math.isfinite(sample.timestamp) and math.isfinite(sample.value)
        ),
        key=lambda point: point[0],
    )
    return (
        [timestamp for timestamp, _ in points],
        [value for _, value in points],
    )


def _last_value(
    timestamps: Sequence[float],
    values: Sequence[float],
    start: float,
    end: float,
) -> float | None:
    left = bisect_left(timestamps, start)
    right = bisect_left(timestamps, end)
    return values[right - 1] if left < right else None


def build_step_frames(
    global_steps: TimeSeries,
    metrics: Sequence[TimeSeries],
    *,
    start_step: int,
    step_count: int,
) -> list[StepFrame]:
    """Build step-indexed snapshots using the last value in each interval."""

    if isinstance(start_step, bool) or not isinstance(start_step, int):
        raise TypeError("start_step must be an integer")
    if isinstance(step_count, bool) or not isinstance(step_count, int):
        raise TypeError("step_count must be an integer")
    if step_count <= 0:
        raise ValueError("step_count must be positive")

    boundaries = _step_boundaries(global_steps)
    required_steps = tuple(range(start_step, start_step + step_count + 1))
    missing = [step for step in required_steps if step not in boundaries]
    if missing:
        values = ", ".join(str(step) for step in missing)
        raise MissingGlobalStepError(f"missing global-step boundaries: {values}")

    interval_times = [boundaries[step] for step in required_steps]
    if any(start >= end for start, end in zip(interval_times, interval_times[1:])):
        raise WindowError("global-step boundaries must have increasing timestamps")

    metric_points: dict[SeriesId, tuple[list[float], list[float]]] = {}
    for metric in metrics:
        if metric.identity == global_steps.identity:
            continue
        if metric.identity in metric_points:
            raise WindowError(f"duplicate metric series: {metric.identity!r}")
        metric_points[metric.identity] = _finite_points(metric)

    frames: list[StepFrame] = []
    for index, step in enumerate(required_steps[:-1]):
        start, end = interval_times[index : index + 2]
        frames.append(
            StepFrame(
                step=step,
                start_time=start,
                end_time=end,
                values={
                    identity: _last_value(timestamps, values, start, end)
                    for identity, (timestamps, values) in metric_points.items()
                },
            )
        )
    return frames


__all__ = [
    "MissingGlobalStepError",
    "StepFrame",
    "WindowError",
    "build_step_frames",
]
