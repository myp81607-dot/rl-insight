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

"""Run baseline fitting, incremental detection, and event diagnosis."""

from __future__ import annotations

import logging
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .association import AssociationItem, AssociationResult, analyze_association
from .baseline import (
    BaselineParameters,
    SeriesBaseline,
    fit_baselines,
)
from .config import DegradationConfig
from .detector import (
    DegradationEvent,
    Direction,
    EventTracker,
    PointResult,
    Policy,
    classify_frame,
)
from .prometheus import GlobalStep, PrometheusClient, Sample, SeriesId, TimeSeries
from .storage import BaselineSnapshot, JsonStorage
from .window import StepFrame, build_step_frames

LOGGER = logging.getLogger(__name__)


class DegradationRuntimeError(RuntimeError):
    """The v0.1 runtime cannot continue with its current state."""


@dataclass(frozen=True)
class RuntimeAssociation:
    """One association result produced by an event transition."""

    phase: str
    event: DegradationEvent
    result: AssociationResult


def _add_observed_boundaries(
    series: TimeSeries, *observations: GlobalStep
) -> TimeSeries:
    samples = {sample.timestamp: sample.value for sample in series.samples}
    for observation in observations:
        samples.setdefault(observation.timestamp, float(observation.value))
    return TimeSeries(
        identity=series.identity,
        samples=tuple(
            Sample(timestamp=timestamp, value=value)
            for timestamp, value in sorted(samples.items())
        ),
    )


def _dominant_direction(
    context: Sequence[Mapping[SeriesId, PointResult]],
    identity: SeriesId,
    *,
    start_step: int,
    end_step: int,
) -> str:
    counts = Counter(
        point.direction.value
        for row in context
        if (point := row[identity]).valid
        and point.abnormal
        and point.direction is not None
        and start_step <= point.step <= end_step
    )
    if not counts:
        return Direction.NORMAL.value
    maximum = max(counts.values())
    leaders = [direction for direction, count in counts.items() if count == maximum]
    return leaders[0] if len(leaders) == 1 else "MIXED"


def _association_directions(
    associations: Sequence[AssociationItem],
    context: Sequence[Mapping[SeriesId, PointResult]],
    event: DegradationEvent,
) -> dict[SeriesId, str]:
    return {
        item.identity: _dominant_direction(
            context,
            item.identity,
            start_step=event.start_step,
            end_step=event.end_step,
        )
        for item in associations
    }


class DegradationRuntime:
    """A synchronous v0.1 degradation workflow."""

    def __init__(
        self,
        client: PrometheusClient,
        config: DegradationConfig,
        *,
        storage: JsonStorage | None = None,
        sleep: Callable[[float], object] = time.sleep,
    ) -> None:
        if not isinstance(config, DegradationConfig):
            raise TypeError("config must be DegradationConfig")
        if not callable(sleep):
            raise TypeError("sleep must be callable")
        self.client = client
        self.config = config
        self.storage = storage or JsonStorage(
            config.standard_data_path,
            config.abnormal_data_path,
        )
        self._sleep = sleep
        history_size = (
            config.runtime.pre_context_steps + config.detector.evidence_window
        )
        self.baselines: dict[SeriesId, SeriesBaseline] = {}
        self.policies: dict[SeriesId, Policy] = {}
        self.target_trackers: dict[SeriesId, EventTracker] = {}
        self._recent_results: deque[dict[SeriesId, PointResult]] = deque(
            maxlen=history_size
        )
        self._active_contexts: dict[SeriesId, list[dict[SeriesId, PointResult]]] = {}
        self._global_step_identity: SeriesId | None = None
        self._next_step: int | None = None
        self._next_start_time: float | None = None
        self._last_processed_step: int | None = None
        self._fitted_parameters: BaselineParameters | None = None
        self._initialization: str | None = None

    @property
    def initialized(self) -> bool:
        """Whether a baseline and a detection cursor are ready."""

        return self._next_step is not None and bool(self.baselines)

    @property
    def next_step(self) -> int | None:
        """The first not-yet-processed complete-step candidate."""

        return self._next_step

    def reset(self) -> None:
        """Clear in-memory state and the two configured JSON files."""

        self.baselines.clear()
        self.policies.clear()
        self.target_trackers.clear()
        self._recent_results.clear()
        self._active_contexts.clear()
        self._global_step_identity = None
        self._next_step = None
        self._next_start_time = None
        self._last_processed_step = None
        self._fitted_parameters = None
        self._initialization = None
        self.storage.reset()

    def initialize(self, *, require_baseline_match: bool = False) -> str:
        """Load a frozen baseline or collect and fit the first 30 steps."""

        if self.initialized and self._initialization is not None:
            return self._initialization
        if self.storage.baseline_exists():
            stored = self.storage.load_baseline()
            expected_count = self.config.runtime.baseline_step_count
            stored_count = stored.end_step - stored.start_step + 1
            mismatch = (
                stored.parameters != self.config.baseline
                or stored_count != expected_count
            )
            if require_baseline_match and mismatch:
                raise DegradationRuntimeError(
                    "the existing baseline used different fit parameters; "
                    "use --reset to refit it"
                )
            if (
                stored.global_step_identity.name
                != self.config.runtime.global_step_metric
            ):
                raise DegradationRuntimeError(
                    "the existing baseline uses a different global-step metric; "
                    "use --reset to refit it"
                )
            self._install_baselines(
                stored.baselines,
                stored.global_step_identity,
                stored.parameters,
            )
            current = self.client.query_global_step(
                self.config.runtime.global_step_metric
            )
            self._next_step = current.value
            self._next_start_time = current.timestamp
            self._initialization = "loaded"
            LOGGER.info("Loaded baseline from %s", self.storage.standard_data_path)
            return self._initialization

        self._fit_initial_baseline()
        self._initialization = "trained"
        return self._initialization

    def _install_baselines(
        self,
        baselines: Mapping[SeriesId, SeriesBaseline],
        global_step_identity: SeriesId,
        parameters: BaselineParameters,
    ) -> None:
        targets = frozenset(self.config.runtime.target_metrics)
        candidates = frozenset(self.config.runtime.candidate_metrics)
        selected = {
            identity: baseline
            for identity, baseline in baselines.items()
            if identity.name in targets or identity.name in candidates
        }
        if not selected:
            raise DegradationRuntimeError(
                "none of the configured target or candidate metrics produced a baseline"
            )
        policies = {
            identity: Policy.UP if identity.name in targets else Policy.BOTH
            for identity in selected
        }
        target_identities = [
            identity for identity, policy in policies.items() if policy is Policy.UP
        ]
        self.baselines = selected
        self.policies = policies
        self.target_trackers = {
            identity: EventTracker(identity, parameters=self.config.detector)
            for identity in target_identities
        }
        self._global_step_identity = global_step_identity
        self._last_processed_step = None
        self._fitted_parameters = parameters

    def _fit_initial_baseline(self) -> None:
        runtime = self.config.runtime
        start = self.client.query_global_step(runtime.global_step_metric)
        required_step = start.value + runtime.baseline_step_count
        current = start
        while current.value < required_step:
            LOGGER.info(
                "Collecting baseline: current step %s, waiting for step %s",
                current.value,
                required_step,
            )
            self._sleep(runtime.poll_interval_seconds)
            current = self.client.query_global_step(runtime.global_step_metric)
            if current.value < start.value:
                raise DegradationRuntimeError("global step moved backwards")

        discovered = self.client.discover_series(
            start=start.timestamp,
            end=current.timestamp,
            selector=runtime.series_selector,
        )
        configured_names = {
            runtime.global_step_metric,
            *runtime.target_metrics,
            *runtime.candidate_metrics,
        }
        discovered = tuple(
            identity for identity in discovered if identity.name in configured_names
        )
        series = self.client.query_range(
            start=start.timestamp,
            end=current.timestamp,
            step=runtime.query_step_seconds,
            query=runtime.series_selector,
            series=discovered,
        )
        global_step_series = _add_observed_boundaries(
            self._find_global_step_series(series), start, current
        )
        frames = build_step_frames(
            global_step_series,
            [item for item in series if item.identity != global_step_series.identity],
            start_step=start.value,
            step_count=runtime.baseline_step_count,
        )
        fitted = fit_baselines(frames, parameters=self.config.baseline)
        if not fitted.baselines:
            raise DegradationRuntimeError("no metric produced a usable baseline")
        self._install_baselines(
            fitted.baselines,
            global_step_series.identity,
            self.config.baseline,
        )
        self.storage.save_baseline(
            BaselineSnapshot(
                start_step=frames[0].step,
                end_step=frames[-1].step,
                global_step_identity=global_step_series.identity,
                parameters=self.config.baseline,
                baselines=self.baselines,
            )
        )
        self._next_step = frames[-1].step + 1
        self._next_start_time = frames[-1].end_time
        LOGGER.info(
            "Trained %s baselines from steps %s-%s; skipped %s series",
            len(fitted.baselines),
            frames[0].step,
            frames[-1].step,
            len(fitted.skipped),
        )

    def _find_global_step_series(self, series: Sequence[TimeSeries]) -> TimeSeries:
        matches = [
            item
            for item in series
            if item.identity.name == self.config.runtime.global_step_metric
        ]
        if len(matches) != 1:
            raise DegradationRuntimeError(
                f"{self.config.runtime.global_step_metric!r} must have exactly one "
                f"range series, got {len(matches)}"
            )
        return matches[0]

    def process_frames(
        self, frames: Sequence[StepFrame]
    ) -> tuple[RuntimeAssociation, ...]:
        """Process already aligned frames in strict global-step order."""

        if not self.baselines:
            raise DegradationRuntimeError("runtime is not initialized")
        expected = (
            None if self._last_processed_step is None else self._last_processed_step + 1
        )
        for frame in frames:
            if expected is not None and frame.step != expected:
                raise DegradationRuntimeError(
                    f"frames must be consecutive: expected {expected}, got {frame.step}"
                )
            expected = frame.step + 1
        produced: list[RuntimeAssociation] = []
        for frame in frames:
            results = classify_frame(frame, self.baselines, self.policies)
            for context in self._active_contexts.values():
                context.append(results)
            self._recent_results.append(results)

            for identity, tracker in sorted(self.target_trackers.items()):
                update = tracker.update(results[identity])
                if update.confirmed_event is not None:
                    event = update.confirmed_event
                    first_step = (
                        event.start_step - self.config.runtime.pre_context_steps
                    )
                    context = [
                        row
                        for row in self._recent_results
                        if row[identity].step >= first_step
                    ]
                    self._active_contexts[identity] = context
                    produced.append(self._analyze_event("confirmed", event, context))
                if update.closed_event is not None:
                    event = update.closed_event
                    context = self._active_contexts.pop(identity)
                    produced.append(self._analyze_event("closed", event, context))
            self._last_processed_step = frame.step
        return tuple(produced)

    def _analyze_event(
        self,
        phase: str,
        event: DegradationEvent,
        context: Sequence[Mapping[SeriesId, PointResult]],
    ) -> RuntimeAssociation:
        target_points = [row[event.identity] for row in context]
        candidates = {
            identity: [row[identity] for row in context]
            for identity, policy in self.policies.items()
            if policy is Policy.BOTH
        }
        result = analyze_association(
            target_points,
            candidates,
            event_start_step=event.start_step,
            parameters=self.config.association,
        )
        top_associations = result.associations[: self.config.runtime.top_k]
        self.storage.save_event(
            phase,
            event,
            result,
            top_associations,
            _association_directions(top_associations, context, event),
        )
        LOGGER.info(
            "%s event for %s at step %s (%s association items)",
            phase.capitalize(),
            event.identity.name,
            event.confirmed_at_step if phase == "confirmed" else event.closed_at_step,
            len(result.associations),
        )
        return RuntimeAssociation(phase=phase, event=event, result=result)

    def run_once(self) -> tuple[RuntimeAssociation, ...]:
        """Fetch and process every newly completed global-step interval."""

        if not self.initialized:
            self.initialize()
        if (
            self._next_step is None
            or self._next_start_time is None
            or self._global_step_identity is None
        ):
            raise DegradationRuntimeError("runtime cursor is not initialized")

        current = self.client.query_global_step(self.config.runtime.global_step_metric)
        step_count = current.value - self._next_step
        if step_count < 0:
            raise DegradationRuntimeError("global step moved backwards")
        if step_count == 0:
            return ()

        identities = (self._global_step_identity, *sorted(self.baselines))
        series = self.client.query_range(
            start=self._next_start_time,
            end=current.timestamp,
            step=self.config.runtime.query_step_seconds,
            query=self.config.runtime.series_selector,
            series=identities,
        )
        global_step_series = next(
            item for item in series if item.identity == self._global_step_identity
        )
        global_step_series = _add_observed_boundaries(
            global_step_series,
            GlobalStep(self._next_start_time, self._next_step),
            current,
        )
        metrics = [item for item in series if item.identity in self.baselines]
        frames = build_step_frames(
            global_step_series,
            metrics,
            start_step=self._next_step,
            step_count=step_count,
        )
        produced = self.process_frames(frames)
        self._next_step = current.value
        self._next_start_time = frames[-1].end_time
        return produced

    def run_forever(self) -> None:
        """Poll forever using the configured interval."""

        if not self.initialized:
            self.initialize()
        while True:
            self.run_once()
            self._sleep(self.config.runtime.poll_interval_seconds)


__all__ = [
    "DegradationRuntime",
    "DegradationRuntimeError",
    "RuntimeAssociation",
]
