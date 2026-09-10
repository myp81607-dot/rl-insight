"""Run the existing degradation algorithm once over offline Prometheus data."""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .association import AssociationParameters, analyze_association
from .baseline import BaselineParameters, SeriesBaseline, fit_baselines
from .detector import (
    DegradationEvent,
    Direction,
    EventTracker,
    PointResult,
    Policy,
    classify_frame,
)
from .input import load_time_series
from .metrics import CANDIDATE_METRICS, GLOBAL_STEP_METRIC, TARGET_METRICS
from .series import SeriesId, TimeSeries
from .state import (
    BaselineSnapshot,
    association_json,
    event_json,
    load_baseline,
    present_events,
    save_baseline,
    save_events,
)
from .window import StepFrame, build_step_frames

BASELINE_STEPS = 30
PRE_CONTEXT_STEPS = 30
TOP_K = 25


class OfflineAnalysisError(RuntimeError):
    """The supplied offline batch cannot complete the analysis."""


def _step_number(value: float) -> int:
    step = round(value)
    if not math.isclose(value, step, rel_tol=0.0, abs_tol=1.0e-9):
        raise OfflineAnalysisError("global-step values must be integers")
    return step


def _global_step_series(series: Sequence[TimeSeries]) -> TimeSeries:
    matches = [item for item in series if item.identity.name == GLOBAL_STEP_METRIC]
    if len(matches) != 1:
        raise OfflineAnalysisError(
            f"{GLOBAL_STEP_METRIC!r} must have exactly one concrete series, "
            f"got {len(matches)}"
        )
    return matches[0]


def _step_span(global_steps: TimeSeries) -> tuple[int, int]:
    """Return the first complete step after range start and the final boundary."""

    if not global_steps.samples:
        raise OfflineAnalysisError("global-step series is empty")
    steps = [_step_number(sample.value) for sample in global_steps.samples]
    return min(steps) + 1, max(steps)


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


class OfflineAnalyzer:
    """Stateful event tracker used only while one offline batch is processed."""

    def __init__(
        self,
        baselines: Mapping[SeriesId, SeriesBaseline],
        *,
        association_parameters: AssociationParameters | None = None,
    ) -> None:
        targets = frozenset(TARGET_METRICS)
        candidates = frozenset(CANDIDATE_METRICS)
        self.baselines = {
            identity: baseline
            for identity, baseline in baselines.items()
            if identity.name in targets or identity.name in candidates
        }
        self.policies = {
            identity: Policy.UP if identity.name in targets else Policy.BOTH
            for identity in self.baselines
        }
        target_ids = [
            identity
            for identity, policy in self.policies.items()
            if policy is Policy.UP
        ]
        candidate_ids = [
            identity
            for identity, policy in self.policies.items()
            if policy is Policy.BOTH
        ]
        if not target_ids:
            raise OfflineAnalysisError("the baseline contains no configured target")
        if not candidate_ids:
            raise OfflineAnalysisError("the baseline contains no configured candidate")
        self.trackers = {identity: EventTracker(identity) for identity in target_ids}
        self.association_parameters = association_parameters or AssociationParameters()
        self.recent: deque[dict[SeriesId, PointResult]] = deque(
            maxlen=PRE_CONTEXT_STEPS + 5
        )
        self.active_contexts: dict[SeriesId, list[dict[SeriesId, PointResult]]] = {}
        self.events: list[dict[str, Any]] = []
        self._last_step: int | None = None

    def _association(
        self,
        event: DegradationEvent,
        context: Sequence[Mapping[SeriesId, PointResult]],
    ) -> dict[str, Any]:
        result = analyze_association(
            [row[event.identity] for row in context],
            {
                identity: [row[identity] for row in context]
                for identity, policy in self.policies.items()
                if policy is Policy.BOTH
            },
            event_start_step=event.start_step,
            parameters=self.association_parameters,
        )
        top = result.associations[:TOP_K]
        directions = {
            item.identity: _dominant_direction(
                context,
                item.identity,
                start_step=event.start_step,
                end_step=event.end_step,
            )
            for item in top
        }
        return association_json(result, top, directions)

    def process(self, frames: Sequence[StepFrame]) -> None:
        for frame in frames:
            if self._last_step is not None and frame.step != self._last_step + 1:
                raise OfflineAnalysisError(
                    f"detection frames must be consecutive: expected "
                    f"{self._last_step + 1}, got {frame.step}"
                )
            results = classify_frame(frame, self.baselines, self.policies)
            for context in self.active_contexts.values():
                context.append(results)
            self.recent.append(results)

            for identity, tracker in sorted(self.trackers.items()):
                update = tracker.update(results[identity])
                if update.confirmed_event is not None:
                    event = update.confirmed_event
                    first_step = event.start_step - PRE_CONTEXT_STEPS
                    context = [
                        row for row in self.recent if row[identity].step >= first_step
                    ]
                    self.active_contexts[identity] = context
                if update.closed_event is not None:
                    event = update.closed_event
                    context = self.active_contexts.pop(identity)
                    self.events.append(
                        event_json(
                            event,
                            self._association(event, context),
                            phase="closed",
                        )
                    )
            self._last_step = frame.step

    def finalize_range_end(self) -> None:
        """Analyze confirmed events that remain open at the selected range end."""

        for identity, tracker in sorted(self.trackers.items()):
            event = tracker.snapshot_open_event()
            if event is None:
                continue
            context = self.active_contexts.pop(identity)
            self.events.append(
                event_json(
                    event,
                    self._association(event, context),
                    phase="open_at_range_end",
                )
            )


def analyze_offline(
    data_dir: Path,
    *,
    start_time: float,
    end_time: float,
    baseline_path: Path,
    output_path: Path,
    promtool_path: Path | None = None,
) -> dict[str, Any]:
    """Read one TSDB range and analyze each final event once."""

    configured_names = {GLOBAL_STEP_METRIC, *TARGET_METRICS, *CANDIDATE_METRICS}
    all_series = load_time_series(
        data_dir,
        start_time=start_time,
        end_time=end_time,
        metric_names=frozenset(configured_names),
        promtool_path=promtool_path,
    )
    selected_series = tuple(
        item for item in all_series if item.identity.name in configured_names
    )
    global_steps = _global_step_series(selected_series)
    first_step, final_boundary = _step_span(global_steps)
    metrics = [
        item for item in selected_series if item.identity != global_steps.identity
    ]

    if baseline_path.exists():
        snapshot = load_baseline(baseline_path)
        baseline_action = "loaded"
    else:
        if final_boundary - first_step < BASELINE_STEPS:
            raise OfflineAnalysisError(
                f"training a baseline requires {BASELINE_STEPS} complete steps; "
                f"the batch contains {final_boundary - first_step}"
            )
        baseline_frames = build_step_frames(
            global_steps,
            metrics,
            start_step=first_step,
            step_count=BASELINE_STEPS,
        )
        fitted = fit_baselines(baseline_frames, parameters=BaselineParameters())
        target_count = sum(
            identity.name in TARGET_METRICS for identity in fitted.baselines
        )
        candidate_count = sum(
            identity.name in CANDIDATE_METRICS for identity in fitted.baselines
        )
        if not target_count or not candidate_count:
            raise OfflineAnalysisError(
                "baseline fitting requires at least one usable configured target "
                "and one usable configured candidate"
            )
        snapshot = BaselineSnapshot(
            start_step=baseline_frames[0].step,
            end_step=baseline_frames[-1].step,
            global_step_identity=global_steps.identity,
            parameters=BaselineParameters(),
            baselines=fitted.baselines,
        )
        save_baseline(baseline_path, snapshot)
        baseline_action = "trained"

    analyzer = OfflineAnalyzer(snapshot.baselines)
    detection_start = max(first_step, snapshot.end_step + 1)
    detection_count = max(0, final_boundary - detection_start)
    if detection_count:
        analysis_metrics = [
            item for item in metrics if item.identity in analyzer.baselines
        ]
        analyzer.process(
            build_step_frames(
                global_steps,
                analysis_metrics,
                start_step=detection_start,
                step_count=detection_count,
            )
        )
        analyzer.finalize_range_end()
    save_events(output_path, analyzer.events)
    return {
        "status": "ok",
        "baseline": {
            "action": baseline_action,
            "path": str(baseline_path.resolve()),
            "start_step": snapshot.start_step,
            "end_step": snapshot.end_step,
            "series_count": len(snapshot.baselines),
        },
        "analysis": {
            "data_dir": str(data_dir.expanduser().resolve()),
            "start_time": start_time,
            "end_time": end_time,
            "detection_start_step": detection_start if detection_count else None,
            "detection_end_step": final_boundary - 1 if detection_count else None,
            "processed_step_count": detection_count,
            "event_count": len(analyzer.events),
            "result_path": str(output_path.resolve()),
        },
        "events": present_events(analyzer.events),
    }


__all__ = ["OfflineAnalysisError", "analyze_offline"]
