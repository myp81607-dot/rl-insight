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

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from experiment.degradation_perception.perception_config import TimeSeries
from experiment.degradation_perception.timeseries import (
    DataValidationError,
    extract_metric_series,
    preprocess_time_series,
    validate_detection_input,
)


METRIC = 'timing_s/step'


@pytest.mark.parametrize(
    ('timestamps', 'values'),
    [
        (None, [1.0]),
        ([1], None),
        (None, None),
    ],
)
def test_preprocess_rejects_none_inputs(timestamps, values):
    with pytest.raises(DataValidationError, match='must not be None'):
        preprocess_time_series(timestamps, values)


def test_preprocess_accepts_two_empty_arrays():
    assert preprocess_time_series([], []) == TimeSeries()


@pytest.mark.parametrize(
    ('timestamps', 'values'),
    [([], [1.0]), ([1], []), ([1, 2], [1.0])],
)
def test_preprocess_rejects_unequal_lengths(timestamps, values):
    with pytest.raises(DataValidationError, match='equal lengths'):
        preprocess_time_series(timestamps, values)


def test_preprocess_filters_invalid_pairs_together():
    series = preprocess_time_series(
        [1, 2, 3, 4, float('inf'), 'bad', True],
        [10, float('nan'), float('inf'), 40, 50, 60, 70],
    )
    assert series == TimeSeries(timestamps=[1.0, 4.0], values=[10.0, 40.0])


def test_preprocess_stable_sorts_and_keeps_last_duplicate():
    series = preprocess_time_series([3, 1, 2, 1], [30, 10, 20, 11])
    assert series == TimeSeries(
        timestamps=[1.0, 2.0, 3.0],
        values=[11.0, 20.0, 30.0],
    )


def test_preprocess_supports_numpy_arrays_and_scalars():
    series = preprocess_time_series(
        np.asarray([np.int64(2), np.int64(1)]),
        np.asarray([np.float64(2.5), np.float32(1.5)]),
    )
    assert series.timestamps == [1.0, 2.0]
    assert series.values == pytest.approx([1.5, 2.5])


def test_preprocess_supports_zero_dimensional_numpy_arrays():
    assert preprocess_time_series(np.asarray(1), np.asarray(2.5)) == TimeSeries(
        timestamps=[1.0],
        values=[2.5],
    )


def test_preprocess_normalizes_datetime_to_utc_epoch_seconds():
    instant = datetime(2026, 1, 1, tzinfo=timezone.utc)
    series = preprocess_time_series(
        [instant, datetime(2026, 1, 1)],
        [1.0, 2.0],
    )
    assert series == TimeSeries(
        timestamps=[instant.timestamp()],
        values=[2.0],
    )


def test_preprocess_drops_nat_and_boolean_values():
    series = preprocess_time_series(
        [np.datetime64('NaT'), 2, 3],
        [1.0, True, 3.0],
    )
    assert series == TimeSeries(timestamps=[3.0], values=[3.0])


def test_detection_input_requires_two_timeseries_phases():
    data = {
        'standard': {
            METRIC: TimeSeries(timestamps=[3, 1, 2], values=[3.0, 1.0, 2.0]),
        },
        'inference': {
            METRIC: TimeSeries(timestamps=[10, 11], values=[1.1, 1.2]),
        },
    }
    validated = validate_detection_input(data)
    assert validated['standard'][METRIC].timestamps == [1.0, 2.0, 3.0]
    assert extract_metric_series(data, 'inference', METRIC).values == [1.1, 1.2]


def test_detection_input_rejects_raw_mapping_and_unknown_phase():
    with pytest.raises(DataValidationError, match='must be a TimeSeries'):
        validate_detection_input(
            {
                'standard': {METRIC: {'timestamps': [1], 'values': [1]}},
                'inference': {},
            }
        )
    with pytest.raises(DataValidationError, match='unsupported root fields'):
        validate_detection_input(
            {'standard': {}, 'inference': {}, 'prometheus': {}}
        )


def test_missing_metric_is_an_empty_series():
    data = {'standard': {}, 'inference': {}}
    assert extract_metric_series(data, 'standard', METRIC) == TimeSeries()
