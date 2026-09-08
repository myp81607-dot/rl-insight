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

"""Pure model-facing presentation of validated degradation state."""

from __future__ import annotations

import copy
from typing import Any

from .metrics import CANDIDATE_CATEGORIES, CATEGORY_BY_METRIC
from .state_schema import StateInspection, StateStatus


class PresentationError(RuntimeError):
    """A requested event or phase is unavailable in valid state."""


def group_top_k(top_k: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group a validated flat Top-K while retaining its global rank."""

    category_order = {
        category: index for index, category in enumerate(CANDIDATE_CATEGORIES)
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for global_rank, raw_item in enumerate(top_k, start=1):
        metric = raw_item["metric"]
        category = CATEGORY_BY_METRIC.get(metric, "uncategorized")
        grouped.setdefault(category, []).append(
            {**copy.deepcopy(raw_item), "global_rank": global_rank}
        )

    def group_key(item: tuple[str, list[dict[str, Any]]]) -> tuple[int, int]:
        category, metrics = item
        return (
            int(metrics[0]["global_rank"]),
            category_order.get(category, len(category_order)),
        )

    return [
        {
            "category": category,
            "distinct_metric_count": len({item["metric"] for item in metrics}),
            "best_global_rank": metrics[0]["global_rank"],
            "metrics": metrics,
        }
        for category, metrics in sorted(grouped.items(), key=group_key)
    ]


def _present_events(
    payload: dict[str, Any], *, event_selection: str, phase_selection: str
) -> dict[str, Any]:
    prepared_events = copy.deepcopy(payload["events"])
    selected_events = (
        prepared_events[-1:] if event_selection == "latest" else prepared_events
    )
    for event in selected_events:
        association = event["association"]
        assert isinstance(association, dict)
        if phase_selection == "auto":
            selected_phases = [
                (
                    "closed"
                    if association.get("closed") is not None
                    else "latest"
                    if association.get("latest") is not None
                    else "confirmed"
                )
            ]
        elif phase_selection == "both":
            selected_phases = [
                phase
                for phase in ("confirmed", "closed")
                if association.get(phase) is not None
            ]
        else:
            if association.get(phase_selection) is None:
                raise PresentationError(
                    f"requested {phase_selection} phase is unavailable for "
                    f"target {event['target']}"
                )
            selected_phases = [phase_selection]

        presented_phases: dict[str, Any] = {}
        for phase in selected_phases:
            phase_payload = copy.deepcopy(association[phase])
            assert isinstance(phase_payload, dict)
            top_k = phase_payload.pop("top_k")
            assert isinstance(top_k, list)
            phase_payload["top_k_by_category"] = group_top_k(top_k)
            presented_phases[phase] = phase_payload
        event["association"] = presented_phases

    result = copy.deepcopy(payload)
    result["events"] = selected_events
    return result


def present_state(
    inspection: StateInspection,
    *,
    kind: str = "all",
    event: str = "latest",
    phase: str = "auto",
) -> dict[str, Any]:
    """Return the stable CLI envelope for one validated inspection."""

    if kind not in {"all", "baseline", "events"}:
        raise ValueError("kind must be all, baseline, or events")
    if event not in {"latest", "all"}:
        raise ValueError("event must be latest or all")
    if phase not in {"auto", "confirmed", "latest", "closed", "both"}:
        raise ValueError("phase must be auto, confirmed, latest, closed, or both")

    if inspection.status is StateStatus.UNINITIALIZED:
        return {"status": inspection.status.value}
    if inspection.status is StateStatus.INVALID_STATE:
        invalid_result = {"status": inspection.status.value}
        if inspection.error is not None:
            invalid_result["error"] = inspection.error
        return invalid_result

    assert inspection.baseline is not None
    assert inspection.events is not None
    result: dict[str, Any] = {"status": inspection.status.value}
    if inspection.status is StateStatus.OK and kind in {"all", "events"}:
        result["selection"] = {"event": event, "phase": phase}
    if kind in {"all", "baseline"}:
        result["baseline"] = copy.deepcopy(inspection.baseline)
    if kind in {"all", "events"}:
        result["events"] = (
            _present_events(
                inspection.events,
                event_selection=event,
                phase_selection=phase,
            )
            if inspection.status is StateStatus.OK
            else copy.deepcopy(inspection.events)
        )
    return result


__all__ = ["PresentationError", "group_top_k", "present_state"]
