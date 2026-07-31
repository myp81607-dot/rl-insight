# Copyright (c) 2026 verl-project authors.
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Validation and normalization for algorithm-owned time-series input.'''

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .perception_config import DetectionInput, TimeSeries

CANONICAL_PHASES = ('standard', 'inference')


class DataValidationError(ValueError):
    '''Raised when input violates the in-memory detection contract.'''


def _sequence_to_list(value: Any, *, field_name: str) -> list[Any]:
    '''Convert supported array-like values without testing ndarray truthiness.'''

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [value.item()]
        return value.tolist()
    if isinstance(value, (str, bytes, bytearray)) or isinstance(value, Mapping):
        raise DataValidationError(f'{field_name} must be an array')
    if not isinstance(value, Sequence):
        raise DataValidationError(f'{field_name} must be an array')
    return list(value)


def _numeric_timestamp(value: Any) -> float:
    '''Convert supported timestamps to one sortable numeric representation.'''

    if isinstance(value, (bool, np.bool_)):
        raise ValueError('boolean is not a timestamp')
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError('timestamp is NaT')
        return float(value.astype('datetime64[ns]').astype(np.int64)) / 1_000_000_000
    if isinstance(value, datetime):
        aware = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return float(aware.timestamp())
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError('timestamp is empty')
        try:
            return float(text)
        except ValueError:
            parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
            aware = (
                parsed
                if parsed.tzinfo is not None
                else parsed.replace(tzinfo=timezone.utc)
            )
            return float(aware.timestamp())
    return float(value)


def _numeric_value(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('boolean is not a metric value')
    return float(value)


def deduplicate_by_timestamp(
    timestamps: Sequence[float],
    values: Sequence[float],
) -> TimeSeries:
    '''Stable-sort aligned pairs and keep the last value per timestamp.'''

    if len(timestamps) != len(values):
        raise DataValidationError(
            'timestamps and values must have equal lengths before deduplication'
        )
    last_values: dict[float, float] = {}
    for timestamp, value in sorted(
        zip(timestamps, values),
        key=lambda item: item[0],
    ):
        last_values[float(timestamp)] = float(value)
    ordered = sorted(last_values.items(), key=lambda item: item[0])
    return TimeSeries(
        timestamps=[item[0] for item in ordered],
        values=[item[1] for item in ordered],
    )


def preprocess_time_series(
    timestamps: Sequence[Any] | np.ndarray | None,
    values: Sequence[Any] | np.ndarray | None,
) -> TimeSeries:
    '''Validate, coerce, filter, sort, and deduplicate one aligned series.'''

    if timestamps is None or values is None:
        raise DataValidationError('timestamps and values must not be None')
    raw_timestamps = _sequence_to_list(timestamps, field_name='timestamps')
    raw_values = _sequence_to_list(values, field_name='values')
    if len(raw_timestamps) != len(raw_values):
        raise DataValidationError(
            'timestamps and values must have equal lengths: '
            f'{len(raw_timestamps)} != {len(raw_values)}'
        )
    if not raw_timestamps:
        return TimeSeries()

    valid_timestamps: list[float] = []
    valid_values: list[float] = []
    for index in range(len(raw_timestamps)):
        try:
            timestamp = _numeric_timestamp(raw_timestamps[index])
            value = _numeric_value(raw_values[index])
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(timestamp) or not math.isfinite(value):
            continue
        valid_timestamps.append(timestamp)
        valid_values.append(value)
    return deduplicate_by_timestamp(valid_timestamps, valid_values)


def standardize_time_series(value: Any, *, location: str) -> TimeSeries:
    '''Validate one TimeSeries supplied by the RL-Insight caller.'''

    if not isinstance(value, TimeSeries):
        raise DataValidationError(f'{location} must be a TimeSeries')
    return preprocess_time_series(value.timestamps, value.values)


def validate_detection_input(payload: Any) -> DetectionInput:
    '''Validate and copy the two-phase in-memory detection input.'''

    if not isinstance(payload, Mapping):
        raise DataValidationError('detection input root must be an object')
    unknown_phases = set(payload) - set(CANONICAL_PHASES)
    missing_phases = set(CANONICAL_PHASES) - set(payload)
    if missing_phases:
        raise DataValidationError(
            f'detection input is missing phases: {sorted(missing_phases)}'
        )
    if unknown_phases:
        raise DataValidationError(
            'detection input contains unsupported root fields: '
            f'{sorted(unknown_phases)}'
        )

    validated: dict[str, dict[str, TimeSeries]] = {}
    for phase in CANONICAL_PHASES:
        section = payload[phase]
        if not isinstance(section, Mapping):
            raise DataValidationError(f'{phase} must be keyed by metric')
        phase_data: dict[str, TimeSeries] = {}
        for metric, value in section.items():
            if not isinstance(metric, str) or not metric:
                raise DataValidationError(
                    f'{phase} metric keys must be non-empty strings'
                )
            phase_data[metric] = standardize_time_series(
                value,
                location=f'{phase}[{metric!r}]',
            )
        validated[phase] = phase_data
    return validated  # type: ignore[return-value]


def extract_metric_series(
    dataset: Mapping[str, Any],
    phase: str,
    metric: str,
) -> TimeSeries:
    '''Extract and validate one standardized metric TimeSeries.'''

    if phase not in CANONICAL_PHASES:
        raise DataValidationError(f'unsupported dataset phase: {phase!r}')
    section = dataset.get(phase)
    if not isinstance(section, Mapping):
        raise DataValidationError(f'{phase} must be keyed by metric')
    if metric not in section:
        return TimeSeries()
    return standardize_time_series(
        section[metric],
        location=f'{phase}[{metric!r}]',
    )


__all__ = [
    'DataValidationError',
    'deduplicate_by_timestamp',
    'extract_metric_series',
    'preprocess_time_series',
    'standardize_time_series',
    'validate_detection_input',
]
