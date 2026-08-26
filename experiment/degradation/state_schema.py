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

"""Pure validation and classification for persisted degradation state."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

SCHEMA_VERSION = 1
NORMAL_RANGE_FIELDS = frozenset(
    {
        "mode_id",
        "start_step",
        "end_step",
        "sample_count",
        "kde_lower",
        "kde_upper",
        "lower",
        "upper",
        "bandwidth",
    }
)
BASELINE_PARAMETER_FIELDS = frozenset(
    {
        "minimum_samples",
        "alpha",
        "lower_ratio",
        "upper_ratio",
        "bandwidth",
        "grid_size",
        "padding_ratio",
        "tail_bandwidths",
        "zero_range_epsilon",
        "random_seed",
        "peak_prominence_ratio",
        "std_factor",
        "within_std_coefficient",
        "minimum_passed_flags",
        "mean_tolerance_ratio",
        "step_gap_factor",
        "maximum_step_gap",
    }
)
NORMAL_RANGE_INTEGER_FIELDS = frozenset(
    {"mode_id", "start_step", "end_step", "sample_count"}
)


class StateStatus(str, Enum):
    """Complete persisted-state outcomes exposed by inspection."""

    OK = "ok"
    NO_ABNORMAL_EVENTS = "no_abnormal_events"
    UNINITIALIZED = "uninitialized"
    INVALID_STATE = "invalid_state"


class StateValidationError(ValueError):
    """A JSON value violates the degradation state contract."""


@dataclass(frozen=True)
class StateInspection:
    """Validated state payloads and their cross-file classification."""

    status: StateStatus
    baseline: dict[str, Any] | None = None
    events: dict[str, Any] | None = None
    error: str | None = None


def empty_events_payload() -> dict[str, Any]:
    """Return the in-memory representation for an absent event file."""

    return {"schema_version": SCHEMA_VERSION, "events": []}


def _valid_identity(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    metric = value.get("metric")
    labels = value.get("labels", {})
    return (
        isinstance(metric, str)
        and bool(metric)
        and isinstance(labels, Mapping)
        and all(
            isinstance(name, str) and isinstance(label, str)
            for name, label in labels.items()
        )
    )


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_schema_version(value: object) -> bool:
    return _is_integer(value) and value == SCHEMA_VERSION


def _finite_number(value: object, *, error: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StateValidationError(error)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StateValidationError(error) from exc
    if not math.isfinite(number):
        raise StateValidationError(error)
    return number


def _parameter_number(
    parameters: Mapping[str, Any], name: str, *, source: str
) -> float | None:
    if name not in parameters:
        return None
    value = parameters[name]
    return _finite_number(value, error=f"{source} has invalid baseline parameters")


def _validate_baseline_parameters(
    parameters: Mapping[str, Any], *, source: str
) -> None:
    if not set(parameters).issubset(BASELINE_PARAMETER_FIELDS):
        raise StateValidationError(f"{source} has invalid baseline parameters")

    for name, minimum in (("minimum_samples", 3), ("grid_size", 32)):
        if name in parameters:
            value = parameters[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise StateValidationError(f"{source} has invalid baseline parameters")
    if "random_seed" in parameters and (
        isinstance(parameters["random_seed"], bool)
        or not isinstance(parameters["random_seed"], int)
    ):
        raise StateValidationError(f"{source} has invalid baseline parameters")
    if "minimum_passed_flags" in parameters:
        value = parameters["minimum_passed_flags"]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 6:
            raise StateValidationError(f"{source} has invalid baseline parameters")

    alpha = _parameter_number(parameters, "alpha", source=source)
    if alpha is not None and not 0.0 < alpha < 0.5:
        raise StateValidationError(f"{source} has invalid baseline parameters")
    for name in ("lower_ratio", "upper_ratio"):
        value = _parameter_number(parameters, name, source=source)
        if value is not None and value < 1.0:
            raise StateValidationError(f"{source} has invalid baseline parameters")
    for name in ("padding_ratio", "peak_prominence_ratio", "mean_tolerance_ratio"):
        value = _parameter_number(parameters, name, source=source)
        if value is not None and value < 0.0:
            raise StateValidationError(f"{source} has invalid baseline parameters")
    for name in (
        "tail_bandwidths",
        "zero_range_epsilon",
        "std_factor",
        "step_gap_factor",
    ):
        value = _parameter_number(parameters, name, source=source)
        if value is not None and value <= 0.0:
            raise StateValidationError(f"{source} has invalid baseline parameters")
    within = _parameter_number(parameters, "within_std_coefficient", source=source)
    if within is not None and not 1.0 <= within < 2.0:
        raise StateValidationError(f"{source} has invalid baseline parameters")
    maximum_gap = parameters.get("maximum_step_gap")
    if maximum_gap is not None:
        value = _parameter_number(parameters, "maximum_step_gap", source=source)
        if value is None or value <= 0.0:
            raise StateValidationError(f"{source} has invalid baseline parameters")
    bandwidth = parameters.get("bandwidth")
    if bandwidth is not None and bandwidth != "auto":
        if isinstance(bandwidth, bool):
            raise StateValidationError(f"{source} has invalid baseline parameters")
        try:
            value = float(bandwidth)
        except (TypeError, ValueError, OverflowError) as exc:
            raise StateValidationError(
                f"{source} has invalid baseline parameters"
            ) from exc
        if not math.isfinite(value) or value <= 0.0:
            raise StateValidationError(f"{source} has invalid baseline parameters")


def validate_baseline_payload(
    payload: Mapping[str, Any], *, source: str = "standard_data.json"
) -> None:
    """Validate one parsed standard_data.json object without I/O."""

    if not _valid_schema_version(payload.get("schema_version")):
        raise StateValidationError(f"{source} has an unsupported schema version")
    baseline = payload.get("baseline")
    if not isinstance(baseline, Mapping):
        raise StateValidationError(f"{source} has an invalid baseline payload")
    try:
        start_step = baseline["start_step"]
        end_step = baseline["end_step"]
        parameters = baseline["parameters"]
        global_step = baseline["global_step"]
        series = baseline["series"]
    except (KeyError, TypeError, ValueError) as exc:
        raise StateValidationError(f"{source} has an invalid baseline payload") from exc
    if (
        not _is_integer(start_step)
        or not _is_integer(end_step)
        or start_step > end_step
        or not isinstance(parameters, Mapping)
        or not _valid_identity(global_step)
        or not isinstance(series, list)
        or not series
    ):
        raise StateValidationError(f"{source} has an invalid baseline payload")
    _validate_baseline_parameters(parameters, source=source)

    for item in series:
        if not _valid_identity(item):
            raise StateValidationError(f"{source} has an invalid baseline series")
        assert isinstance(item, Mapping)
        try:
            sample_count = item["sample_count"]
            ranges = item["ranges"]
        except (KeyError, TypeError, ValueError) as exc:
            raise StateValidationError(
                f"{source} has an invalid baseline series"
            ) from exc
        if (
            not _is_integer(sample_count)
            or sample_count <= 0
            or not isinstance(ranges, list)
            or not ranges
        ):
            raise StateValidationError(f"{source} has an invalid baseline series")
        for normal_range in ranges:
            if not isinstance(normal_range, Mapping) or set(normal_range) != set(
                NORMAL_RANGE_FIELDS
            ):
                raise StateValidationError(f"{source} has an invalid normal range")
            if any(
                not _is_integer(normal_range[field])
                for field in NORMAL_RANGE_INTEGER_FIELDS
            ):
                raise StateValidationError(f"{source} has an invalid normal range")
            for field in NORMAL_RANGE_FIELDS - NORMAL_RANGE_INTEGER_FIELDS:
                _finite_number(
                    normal_range[field],
                    error=f"{source} has an invalid normal range",
                )


def _validate_top_k(top_k: object, *, source: str) -> None:
    if not isinstance(top_k, list):
        raise StateValidationError(f"{source} association top_k must be a list")
    previous_score = math.inf
    for raw_item in top_k:
        if not isinstance(raw_item, Mapping):
            raise StateValidationError(f"{source} association item must be an object")
        metric = raw_item.get("metric")
        labels = raw_item.get("labels")
        direction = raw_item.get("direction")
        score = raw_item.get("association_percent")
        if not isinstance(metric, str) or not metric:
            raise StateValidationError(
                f"{source} association item has an invalid metric"
            )
        if not isinstance(labels, Mapping) or any(
            not isinstance(name, str) or not isinstance(label, str)
            for name, label in labels.items()
        ):
            raise StateValidationError(
                f"{source} association item for {metric} has invalid labels"
            )
        if not isinstance(direction, str) or not direction:
            raise StateValidationError(
                f"{source} association item for {metric} has an invalid direction"
            )
        score_error = f"{source} association item for {metric} has an invalid score"
        score_number = _finite_number(score, error=score_error)
        if (
            score_number < -1.0e-9
            or score_number > 100.0 + 1.0e-9
            or score_number > previous_score + 1.0e-9
        ):
            raise StateValidationError(score_error)
        previous_score = score_number


def validate_events_payload(
    payload: Mapping[str, Any], *, source: str = "abnormal_data.json"
) -> None:
    """Validate one parsed abnormal_data.json object without I/O."""

    if not _valid_schema_version(payload.get("schema_version")):
        raise StateValidationError(f"{source} has an unsupported schema version")
    events = payload.get("events")
    if not isinstance(events, list):
        raise StateValidationError(f"{source} has an invalid events payload")
    for event in events:
        if not isinstance(event, Mapping):
            raise StateValidationError(f"{source} contains an invalid event")
        if not isinstance(event.get("target"), str) or not event["target"]:
            raise StateValidationError(f"{source} contains an invalid target")
        target_labels = event.get("target_labels")
        if not isinstance(target_labels, Mapping) or any(
            not isinstance(name, str) or not isinstance(label, str)
            for name, label in target_labels.items()
        ):
            raise StateValidationError(f"{source} contains invalid target labels")
        if not isinstance(event.get("direction"), str) or not event["direction"]:
            raise StateValidationError(f"{source} contains an invalid direction")
        for field in ("start_step", "end_step", "confirmed_at_step", "abnormal_points"):
            value = event.get(field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise StateValidationError(f"{source} contains an invalid {field}")
        for field in ("start_time", "end_time", "confirmed_at_time"):
            _finite_number(
                event.get(field), error=f"{source} contains an invalid {field}"
            )

        association = event.get("association")
        if not isinstance(association, Mapping):
            raise StateValidationError(f"{source} event association must be an object")
        if "closed_at_step" not in event or "closed_at_time" not in event:
            raise StateValidationError(
                f"{source} event is missing closed lifecycle fields"
            )
        if "confirmed" not in association or "closed" not in association:
            raise StateValidationError(
                f"{source} event association is missing a phase field"
            )
        closed_step = event.get("closed_at_step")
        closed_time = event.get("closed_at_time")
        closed_association = association.get("closed")
        if closed_association is None:
            if closed_step is not None or closed_time is not None:
                raise StateValidationError(
                    f"{source} open event has inconsistent closed fields"
                )
        else:
            closed_error = f"{source} closed event has inconsistent closed fields"
            if not _is_integer(closed_step):
                raise StateValidationError(closed_error)
            _finite_number(closed_time, error=closed_error)
        for phase in ("confirmed", "closed"):
            phase_payload = association.get(phase)
            if phase_payload is None:
                if phase == "confirmed":
                    raise StateValidationError(
                        f"{source} event association confirmed phase must be an object"
                    )
                continue
            if not isinstance(phase_payload, Mapping):
                raise StateValidationError(
                    f"{source} event association {phase} must be an object"
                )
            _validate_top_k(phase_payload.get("top_k"), source=source)


def classify_state(
    *,
    baseline_present: bool,
    baseline_payload: dict[str, Any] | None,
    events_present: bool,
    events_payload: dict[str, Any] | None,
    baseline_source: str = "standard_data.json",
    events_source: str = "abnormal_data.json",
) -> StateInspection:
    """Validate parsed payloads and classify their cross-file state."""

    if not baseline_present and not events_present:
        return StateInspection(StateStatus.UNINITIALIZED)
    if not baseline_present:
        return StateInspection(
            StateStatus.INVALID_STATE,
            error=f"{events_source} exists without {baseline_source}",
        )
    if baseline_payload is None:
        return StateInspection(
            StateStatus.INVALID_STATE,
            error=f"{baseline_source} is missing its parsed payload",
        )
    try:
        validate_baseline_payload(baseline_payload, source=baseline_source)
        if events_present:
            if events_payload is None:
                raise StateValidationError(
                    f"{events_source} is missing its parsed payload"
                )
            validate_events_payload(events_payload, source=events_source)
        else:
            events_payload = empty_events_payload()
    except StateValidationError as exc:
        return StateInspection(StateStatus.INVALID_STATE, error=str(exc))

    assert events_payload is not None
    status = (
        StateStatus.OK if events_payload["events"] else StateStatus.NO_ABNORMAL_EVENTS
    )
    return StateInspection(
        status,
        baseline=baseline_payload,
        events=events_payload,
    )


__all__ = [
    "SCHEMA_VERSION",
    "StateInspection",
    "StateStatus",
    "StateValidationError",
    "classify_state",
    "empty_events_payload",
    "validate_baseline_payload",
    "validate_events_payload",
]
