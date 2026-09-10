"""Minimal JSON persistence for an offline baseline and analysis result."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .association import AssociationItem, AssociationResult
from .baseline import BaselineParameters, NormalRange, SeriesBaseline
from .detector import DegradationEvent
from .metrics import CANDIDATE_CATEGORIES, CATEGORY_BY_METRIC
from .series import SeriesId

SCHEMA_VERSION = 1


class StateError(RuntimeError):
    """A baseline or result file cannot be read or written."""


@dataclass(frozen=True)
class BaselineSnapshot:
    start_step: int
    end_step: int
    global_step_identity: SeriesId
    parameters: BaselineParameters
    baselines: dict[SeriesId, SeriesBaseline]


def _identity_json(identity: SeriesId) -> dict[str, Any]:
    return {"metric": identity.name, "labels": dict(identity.labels)}


def _identity_from_json(value: object) -> SeriesId:
    if not isinstance(value, Mapping):
        raise StateError("series identity must be an object")
    metric = value.get("metric")
    labels = value.get("labels", {})
    if not isinstance(metric, str) or not metric or not isinstance(labels, Mapping):
        raise StateError("series identity is invalid")
    label_set: dict[str, str] = {"__name__": metric}
    for key, label_value in labels.items():
        if not isinstance(key, str) or not isinstance(label_value, str):
            raise StateError("series labels must be strings")
        label_set[key] = label_value
    return SeriesId.from_label_set(label_set)


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise StateError(f"cannot write {path}: {exc}") from exc


def save_baseline(path: Path, snapshot: BaselineSnapshot) -> None:
    series = []
    for identity, baseline in sorted(snapshot.baselines.items()):
        series.append(
            {
                **_identity_json(identity),
                "sample_count": baseline.sample_count,
                "ranges": [asdict(normal_range) for normal_range in baseline.ranges],
            }
        )
    _write(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "baseline": {
                "start_step": snapshot.start_step,
                "end_step": snapshot.end_step,
                "global_step": _identity_json(snapshot.global_step_identity),
                "parameters": asdict(snapshot.parameters),
                "series": series,
            },
        },
    )


def load_baseline(path: Path) -> BaselineSnapshot:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload["baseline"]
        parameters = BaselineParameters(**raw["parameters"])
        baselines: dict[SeriesId, SeriesBaseline] = {}
        for item in raw["series"]:
            identity = _identity_from_json(item)
            baselines[identity] = SeriesBaseline(
                identity=identity,
                sample_count=int(item["sample_count"]),
                ranges=tuple(NormalRange(**value) for value in item["ranges"]),
            )
        snapshot = BaselineSnapshot(
            start_step=int(raw["start_step"]),
            end_step=int(raw["end_step"]),
            global_step_identity=_identity_from_json(raw["global_step"]),
            parameters=parameters,
            baselines=baselines,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise StateError(f"cannot load baseline {path}: {exc}") from exc
    if payload.get("schema_version") != SCHEMA_VERSION or not snapshot.baselines:
        raise StateError(f"baseline {path} has an unsupported or empty schema")
    return snapshot


def association_json(
    result: AssociationResult,
    top_associations: Sequence[AssociationItem],
    directions: Mapping[SeriesId, str],
) -> dict[str, Any]:
    return {
        "status": result.status,
        "top_k": [
            {
                **_identity_json(item.identity),
                "direction": directions[item.identity],
                "association_percent": item.association_score * 100.0,
                "pearson": item.pearson,
                "spearman": item.spearman,
                "selected_correlation": item.selected_correlation,
                "selected_correlation_method": item.selected_correlation_method,
                "correlation_share": item.correlation_share,
                "random_forest_eligible": item.random_forest_eligible,
                "random_forest_importance": item.random_forest_importance,
                "random_forest_share": item.random_forest_share,
                "aligned_sample_count": item.aligned_sample_count,
                "coverage_ratio": item.coverage_ratio,
            }
            for item in top_associations
        ],
        "effective_correlation_weight": result.effective_correlation_weight,
        "effective_random_forest_weight": result.effective_random_forest_weight,
        "random_forest_status": result.random_forest_status,
        "random_forest_reason": result.random_forest_reason,
        "random_forest_sample_count": result.random_forest_sample_count,
        "random_forest_validation_score": result.random_forest_validation_score,
        "skipped_candidates": [
            {**_identity_json(item.identity), "reason": item.reason}
            for item in result.skipped_candidates
        ],
    }


def event_json(
    event: DegradationEvent,
    association: Mapping[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    """Serialize one final event view and its single whole-event analysis."""

    if phase not in {"closed", "open_at_range_end"}:
        raise ValueError("phase must be closed or open_at_range_end")

    return {
        "target": event.identity.name,
        "target_labels": dict(event.identity.labels),
        "direction": event.direction.value,
        "start_step": event.start_step,
        "start_time": event.start_time,
        "end_step": event.end_step,
        "end_time": event.end_time,
        "confirmed_at_step": event.confirmed_at_step,
        "confirmed_at_time": event.confirmed_at_time,
        "closed_at_step": event.closed_at_step,
        "closed_at_time": event.closed_at_time,
        "abnormal_points": event.abnormal_points,
        "event_state": phase,
        "association": {phase: dict(association)},
    }


def save_events(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    _write(path, {"schema_version": SCHEMA_VERSION, "events": list(events)})


def _group_top_k(top_k: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    category_order = {name: index for index, name in enumerate(CANDIDATE_CATEGORIES)}
    groups: dict[str, list[dict[str, Any]]] = {}
    for rank, raw in enumerate(top_k, start=1):
        item = dict(raw)
        item["global_rank"] = rank
        category = CATEGORY_BY_METRIC.get(str(item["metric"]), "uncategorized")
        groups.setdefault(category, []).append(item)
    ordered = sorted(
        groups.items(),
        key=lambda pair: (
            pair[1][0]["global_rank"],
            category_order.get(pair[0], len(category_order)),
        ),
    )
    return [
        {
            "category": category,
            "distinct_metric_count": len({item["metric"] for item in metrics}),
            "best_global_rank": metrics[0]["global_rank"],
            "metrics": metrics,
        }
        for category, metrics in ordered
    ]


def present_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Present closed and range-end-open event association results."""

    presented = []
    for event in json.loads(json.dumps(list(events))):
        phase = str(event.get("event_state", "closed"))
        association = event["association"].get(phase)
        if association is None:
            continue
        association["top_k_by_category"] = _group_top_k(association.pop("top_k"))
        event["association"] = {phase: association}
        presented.append(event)
    return presented


__all__ = [
    "BaselineSnapshot",
    "StateError",
    "association_json",
    "event_json",
    "load_baseline",
    "present_events",
    "save_baseline",
    "save_events",
]
