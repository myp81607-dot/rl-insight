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

import json
from types import SimpleNamespace

import pytest

from experiment.degradation_perception.detection_runtime import (
    DetectionRunner,
    build_detection_dataset,
    build_kde_debug_details,
    format_terminal_summary,
    serialize_result,
)
from experiment.degradation_perception.perception_config import (
    ThresholdModel,
    TimeSeries,
)


METRIC = "timing_s/step"
STANDARD_VALUES = [
    1.00,
    1.01,
    0.99,
    1.02,
    1.00,
    0.98,
    1.01,
    1.00,
    0.99,
    1.02,
] * 3
NORMAL_INFERENCE = [1.00, 1.01, 0.99, 1.00, 1.02, 0.98]


def _series(start: int, values: list[float]) -> TimeSeries:
    return TimeSeries(
        timestamps=list(range(start, start + len(values))),
        values=list(values),
    )


def _dataset(
    standard: list[float] | None = None,
    inference: list[float] | None = None,
) -> dict[str, dict[str, TimeSeries]]:
    return {
        "standard": {METRIC: _series(1, standard or STANDARD_VALUES)},
        "inference": {METRIC: _series(100, inference or NORMAL_INFERENCE)},
    }


def _fake_detector(*, standard_models=None):
    return SimpleNamespace(
        config_dict={
            METRIC: {
                "alpha": 0.025,
                "lower_ratio": 1.10,
                "upper_ratio": 1.20,
                "kde": {"bandwidth": "auto"},
            }
        },
        metric_policies={},
        standard_models=standard_models or {},
    )


def test_build_detection_dataset_never_copies_between_phases():
    standard_metric = "standard-only"
    inference_metric = "inference-only"
    standard_series = _series(1, [1.0, 1.1, 1.2])
    inference_series = _series(100, [9.0, 9.1])

    dataset = build_detection_dataset(
        {standard_metric: standard_series},
        {inference_metric: inference_series},
        [standard_metric, inference_metric],
    )

    assert dataset["standard"][standard_metric] == standard_series
    assert dataset["standard"][inference_metric] == TimeSeries()
    assert dataset["inference"][standard_metric] == TimeSeries()
    assert dataset["inference"][inference_metric] == inference_series
    assert dataset["standard"] is not dataset["inference"]


def test_real_runner_exposes_complete_kde_debug_and_normal_success():
    runner = DetectionRunner(
        standard={METRIC: _series(1, STANDARD_VALUES)},
        metrics=METRIC,
        task_id=None,
        source_type="prometheus",
    )

    result = runner.run({METRIC: _series(100, NORMAL_INFERENCE)})
    details = result["debugDetails"]["metrics"][METRIC]

    assert result["taskId"] == "default"
    assert result["states"] == {METRIC: 0}
    assert result["abnormalTimeRange"][METRIC] == []
    assert result["results"][METRIC]["currentAbnormalTimeRange"] == []
    assert details["state"] == 0
    assert details["standardSamples"] == len(STANDARD_VALUES)
    assert details["inferenceSamples"] == len(NORMAL_INFERENCE)
    assert details["alpha"] == pytest.approx(0.01)
    assert details["lowerProbability"] == pytest.approx(0.01)
    assert details["upperProbability"] == pytest.approx(0.99)
    assert details["normalModelCount"] == len(details["normalModels"])
    assert details["normalModelCount"] > 0
    assert details["modeledStandardSamples"] == sum(
        model["standardSegmentSamples"] for model in details["normalModels"]
    )
    assert details["abnormalPointCount"] == 0
    assert details["abnormalIntervalCount"] == 0

    required_model_fields = {
        "standardSegmentSamples",
        "rawLowerThreshold",
        "rawUpperThreshold",
        "finalLowerThreshold",
        "finalUpperThreshold",
        "effectiveBandwidth",
    }
    for model in details["normalModels"]:
        assert required_model_fields <= model.keys()
        assert model["standardSegmentSamples"] > 0
        assert model["rawLowerThreshold"] is not None
        assert model["rawUpperThreshold"] is not None
        assert model["finalLowerThreshold"] is not None
        assert model["finalUpperThreshold"] is not None
        assert model["effectiveBandwidth"] is not None

    json.dumps(result, allow_nan=False)
    assert json.loads(serialize_result(result)) == result


def test_build_kde_debug_details_preserves_each_normal_model():
    models = [
        ThresholdModel(
            mode_id=3,
            segment_start_time=10.0,
            segment_end_time=12.0,
            point_count=3,
            lower_kde_threshold=0.81,
            upper_kde_threshold=1.19,
            lower_threshold=0.73,
            upper_threshold=1.43,
            bandwidth=0.11,
            diagnostics={"influenceRegion": [0.7, 1.3]},
        ),
        ThresholdModel(
            mode_id=7,
            segment_start_time=20.0,
            segment_end_time=24.0,
            point_count=5,
            lower_kde_threshold=4.82,
            upper_kde_threshold=5.18,
            lower_threshold=4.38,
            upper_threshold=6.22,
            bandwidth=0.22,
            diagnostics={"influenceRegion": [4.7, 5.3]},
        ),
    ]
    result = {
        "states": {METRIC: 0},
        "results": {
            METRIC: {
                "thresholds": models,
                "pointDiagnostics": [
                    {"abnormal": False},
                    {"abnormal": True},
                ],
                "currentAbnormalTimeRange": [],
            }
        },
        "abnormalTimeRange": {METRIC: []},
    }

    details = build_kde_debug_details(
        _fake_detector(),
        result,
        _dataset(),
        METRIC,
    )[METRIC]

    assert details["normalModelCount"] == 2
    assert details["modeledStandardSamples"] == 8
    assert [model["modelId"] for model in details["normalModels"]] == [3, 7]
    assert [
        model["rawLowerThreshold"] for model in details["normalModels"]
    ] == [0.81, 4.82]
    assert [
        model["finalUpperThreshold"] for model in details["normalModels"]
    ] == [1.43, 6.22]
    assert [
        model["effectiveBandwidth"] for model in details["normalModels"]
    ] == [0.11, 0.22]
    assert details["abnormalPointCount"] == 1


@pytest.mark.parametrize("state", [1, 2])
def test_nonzero_state_never_reports_an_abnormal_point_count(state):
    result = {
        "states": {METRIC: state},
        "results": {
            METRIC: {
                "thresholds": [],
                "pointDiagnostics": [{"abnormal": True}],
                "currentAbnormalTimeRange": [],
            }
        },
        "abnormalTimeRange": {METRIC: []},
    }

    details = build_kde_debug_details(
        _fake_detector(),
        result,
        _dataset(),
        METRIC,
    )[METRIC]

    assert details["state"] == state
    assert details["abnormalPointCount"] is None


def test_terminal_rounding_is_display_only_and_json_keeps_precision():
    alpha = 0.123456789012345
    result = {
        "taskId": "precision-check",
        "states": {METRIC: 0},
        "debugDetails": {
            "metrics": {
                METRIC: {
                    "state": 0,
                    "standardSamples": 30,
                    "inferenceSamples": 6,
                    "alpha": alpha,
                    "lowerProbability": alpha,
                    "upperProbability": 0.876543210987655,
                    "normalModelCount": 1,
                    "normalModels": [
                        {
                            "rawLowerThreshold": 1.23456789012345,
                            "rawUpperThreshold": 9.87654321098765,
                            "finalLowerThreshold": 1.11111111111111,
                            "finalUpperThreshold": 10.2222222222222,
                            "lowerRatio": 1.12345678901234,
                            "upperRatio": 1.23456789012345,
                            "effectiveBandwidth": 0.333333333333333,
                        }
                    ],
                    "abnormalPointCount": 0,
                    "abnormalIntervalCount": 0,
                }
            }
        },
    }

    terminal = format_terminal_summary(result, debug_kde=True)
    assert "Alpha: 0.12" in terminal
    assert "Lower probability: 0.12" in terminal
    assert "Upper probability: 0.88" in terminal
    assert "KDE standard range: 1.11 ~ 10.22" in terminal
    assert "Raw KDE range: 1.23 ~ 9.88" in terminal
    assert "Lower ratio: 1.12" in terminal
    assert "Upper ratio: 1.23" in terminal
    assert "KDE bandwidth: 0.33" in terminal

    encoded = serialize_result(result)
    decoded = json.loads(encoded)
    assert str(alpha) in encoded
    assert decoded["debugDetails"]["metrics"][METRIC]["alpha"] == alpha
    assert (
        decoded["debugDetails"]["metrics"][METRIC]["normalModels"][0]
        ["rawLowerThreshold"]
        == 1.23456789012345
    )
