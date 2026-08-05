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

from __future__ import annotations

import pytest

from experiment.degradation_perception.perception_config import TimeSeries
from experiment.degradation_perception.prometheus_source import (
    PrometheusDataError,
    PrometheusHttpClient,
    build_bootstrap_window,
    filter_after_global_step,
)


class FakeResponse:
    def __init__(self, payload, *, status_error: Exception | None = None):
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error is not None:
            raise self.status_error

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def _matrix(*series):
    return {
        "status": "success",
        "data": {"resultType": "matrix", "result": list(series)},
    }


def test_query_range_parses_real_matrix_shape_and_deduplicates():
    session = FakeSession(
        FakeResponse(
            _matrix(
                {
                    "metric": {"__name__": "timing", "instance": "trainer-0"},
                    "values": [
                        [1000, "1.0"],
                        [1001, "NaN"],
                        [1000, "2.0"],
                        [1002, "3.0"],
                    ],
                }
            )
        )
    )
    client = PrometheusHttpClient("http://prometheus:9090/", session=session)

    series = client.query_range("timing_metric", start=1000, end=1010, step=15)

    assert series == TimeSeries(timestamps=[1000.0, 1002.0], values=[2.0, 3.0])
    url, kwargs = session.calls[0]
    assert url == "http://prometheus:9090/api/v1/query_range"
    assert kwargs["params"] == {
        "query": "timing_metric",
        "start": 1000.0,
        "end": 1010.0,
        "step": 15.0,
    }


def test_query_range_rejects_ambiguous_multiple_series():
    session = FakeSession(
        FakeResponse(
            _matrix(
                {"metric": {"instance": "a"}, "values": [[1, "1"]]},
                {"metric": {"instance": "b"}, "values": [[1, "2"]]},
            )
        )
    )
    client = PrometheusHttpClient("http://prometheus:9090", session=session)

    with pytest.raises(PrometheusDataError, match="returned 2 series"):
        client.query_range("timing_metric", start=1, end=2, step=1)


def test_bootstrap_uses_one_latest_sample_for_each_distinct_step():
    global_steps = TimeSeries(
        timestamps=[10, 11, 20, 30, 40],
        values=[5, 5, 6, 7, 8],
    )
    metrics = {
        "timing_s/step": TimeSeries(
            timestamps=[11, 20.2, 30.1, 50],
            values=[1.0, 1.1, 1.2, 9.9],
        )
    }

    window = build_bootstrap_window(
        global_steps,
        metrics,
        start_step=5,
        end_step=7,
        alignment_tolerance_seconds=0.5,
    )

    assert window.observed_steps == (5, 6, 7)
    assert window.start_timestamp == 11
    assert window.end_timestamp == 30
    assert window.series["timing_s/step"] == TimeSeries(
        timestamps=[5.0, 6.0, 7.0],
        values=[1.0, 1.1, 1.2],
    )


def test_filter_after_global_step_keeps_wall_clock_timestamps():
    global_steps = TimeSeries(
        timestamps=[100, 110, 120, 130],
        values=[24, 25, 26, 27],
    )
    metrics = {
        "timing_s/step": TimeSeries(
            timestamps=[100.1, 110.1, 120.1, 130.1],
            values=[1.0, 1.1, 2.0, 2.1],
        )
    }

    filtered = filter_after_global_step(
        global_steps,
        metrics,
        minimum_step_exclusive=25,
        alignment_tolerance_seconds=1,
    )

    assert filtered["timing_s/step"] == TimeSeries(
        timestamps=[120.1, 130.1],
        values=[2.0, 2.1],
    )
