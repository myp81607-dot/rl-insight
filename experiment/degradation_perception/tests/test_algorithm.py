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

import pytest

from experiment.degradation_perception.algorithm import (
    DegradationPerception,
    _classify_value,
    build_standard_data,
    get_standard_data,
)
from experiment.degradation_perception.perception_config import (
    DetectionInput,
    DetectionResult,
    ThresholdModel,
    TimeSeries,
)
from experiment.degradation_perception.policy import resolve_metric_policy


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
UP_INFERENCE = [1.00, 1.01, 0.99, 1.00, 1.50, 1.51, 1.49, 1.50, 1.52, 1.50]


def make_dataset(
    standard_values=STANDARD_VALUES,
    inference_values=NORMAL_INFERENCE,
    *,
    metric=METRIC,
):
    data: DetectionInput = {
        "standard": {
            metric: TimeSeries(
                timestamps=list(range(1, len(standard_values) + 1)),
                values=list(standard_values),
            )
        },
        "inference": {
            metric: TimeSeries(
                timestamps=list(range(100, 100 + len(inference_values))),
                values=list(inference_values),
            )
        },
    }
    return data


def detector(_tmp_path, dataset, **kwargs):
    return DegradationPerception(
        dataset=dataset,
        metrics=kwargs.pop("metrics", [METRIC]),
        **kwargs,
    )


def test_get_standard_data_returns_aligned_sorted_series():
    dataset = make_dataset([1.0, 1.1, 1.2])
    dataset["standard"][METRIC] = TimeSeries(
        timestamps=[3, 1, 2, 1],
        values=[3.0, 1.0, 2.0, 1.1],
    )
    series = get_standard_data(dataset, METRIC)
    assert series.timestamps == [1.0, 2.0, 3.0]
    assert series.values == [1.1, 2.0, 3.0]


def test_build_standard_data_applies_one_minus_alpha_and_outward_ratios():
    config = resolve_metric_policy(METRIC)
    models = build_standard_data(
        list(range(len(STANDARD_VALUES))),
        STANDARD_VALUES,
        config,
    )
    assert models
    for model in models:
        assert model.upper_threshold == pytest.approx(
            model.upper_kde_threshold * config["upper_ratio"]
        )
        assert model.lower_threshold == pytest.approx(
            model.lower_kde_threshold / config["lower_ratio"]
        )
        assert (
            model.lower_threshold
            <= model.lower_kde_threshold
            < model.upper_kde_threshold
            <= model.upper_threshold
        )


def test_task_id_none_and_normal_response_are_json_serializable(tmp_path):
    response = detector(tmp_path, make_dataset(), task_id=None).detect()
    assert response["taskId"] == "default"
    assert response["states"] == {METRIC: 0}
    assert response["abnormalTimeRange"][METRIC] == []
    json.dumps(response, allow_nan=False)


def test_standard_insufficient_has_state_one_and_no_fabricated_detection(tmp_path):
    response = detector(tmp_path, make_dataset([1.0, 1.1], UP_INFERENCE)).detect()
    assert response["states"][METRIC] == 1
    assert response["results"][METRIC]["thresholds"] == []
    assert response["results"][METRIC]["abnormalTimeRange"] == []


def test_standard_state_one_has_priority_when_both_phases_are_insufficient(tmp_path):
    response = detector(tmp_path, make_dataset([1.0], [1.0])).detect()
    assert response["states"][METRIC] == 1


def test_inference_insufficient_has_state_two_and_no_degradation(tmp_path):
    response = detector(tmp_path, make_dataset(STANDARD_VALUES, [1.5] * 4)).detect()
    assert response["states"][METRIC] == 2
    assert response["results"][METRIC]["thresholds"] == []
    assert response["abnormalTimeRange"][METRIC] == []


def test_sustained_up_degradation_produces_a_formal_interval(tmp_path):
    response = detector(tmp_path, make_dataset(STANDARD_VALUES, UP_INFERENCE)).detect()
    assert response["states"][METRIC] == 0
    intervals = response["abnormalTimeRange"][METRIC]
    assert len(intervals) == 1
    interval = intervals[0]
    assert interval["abnormalPointCount"] >= 5
    assert interval["abnormalRate"] > 0.60
    assert interval["duration"] > 0.5
    assert interval["abnormalType"] == "UP"
    assert interval["validationDetail"] == {
        "condition_1": True,
        "condition_2": True,
        "condition_3": True,
        "condition_4": True,
    }


@pytest.mark.parametrize(
    ("abnormal_type", "inference_values"),
    [
        ("UP", [1.5] * 6),
        ("DOWN", [0.5] * 6),
        ("BOTH", [1.5] * 6),
    ],
)
def test_configured_abnormal_types_are_not_inferred_from_metric_name(
    abnormal_type, inference_values
):
    response = DegradationPerception(
        dataset=make_dataset(STANDARD_VALUES, inference_values),
        metrics=[METRIC],
        metric_policies={METRIC: {"abnormal_type": abnormal_type}},
    ).detect()
    assert response["states"][METRIC] == 0
    assert response["abnormalTimeRange"][METRIC][0]["abnormalType"] == abnormal_type


@pytest.mark.parametrize(
    ("standard_values", "inference_values", "abnormal_type"),
    [
        (STANDARD_VALUES, [1.0] * 6, "DOWN"),
        (STANDARD_VALUES, [1.0] * 6, "BOTH"),
        ([-value for value in STANDARD_VALUES], [-1.0] * 6, "UP"),
        ([-value for value in STANDARD_VALUES], [-1.0] * 6, "BOTH"),
    ],
)
def test_outward_thresholds_do_not_flag_normal_positive_or_negative_data(
    standard_values, inference_values, abnormal_type
):
    response = DegradationPerception(
        dataset=make_dataset(standard_values, inference_values),
        metrics=[METRIC],
        metric_policies={METRIC: {"abnormal_type": abnormal_type}},
    ).detect()
    assert response["states"][METRIC] == 0
    assert response["abnormalTimeRange"][METRIC] == []
    assert not any(
        item["abnormal"] for item in response["results"][METRIC]["pointDiagnostics"]
    )


def test_epoch_timeseries_runs_end_to_end_and_preserves_seconds():
    metric = "rl_insight_monitor_timing_s_step"
    response = DegradationPerception(
        dataset={
            "standard": {
                metric: TimeSeries(
                    timestamps=[
                        1_710_000_000 + index * 15
                        for index in range(len(STANDARD_VALUES))
                    ],
                    values=list(STANDARD_VALUES),
                )
            },
            "inference": {
                metric: TimeSeries(
                    timestamps=[
                        1_710_001_000 + index * 15
                        for index in range(len(UP_INFERENCE))
                    ],
                    values=list(UP_INFERENCE),
                )
            },
        },
        metrics=[metric],
        source_type="prometheus",
    ).detect()
    assert response["states"][metric] == 0
    interval = response["abnormalTimeRange"][metric][0]
    assert interval["startTime"] > 1_700_000_000
    assert interval["endTime"] > interval["startTime"]
def test_known_high_normal_mode_is_not_flagged_by_up_detection(tmp_path):
    standard = [1.00, 1.01, 1.02, 1.04, 1.05, 5.00, 5.01, 5.02]
    response = detector(
        tmp_path,
        make_dataset(standard, [5.00, 5.01, 5.02, 5.01, 5.00, 5.02]),
    ).detect()
    assert response["states"][METRIC] == 0
    assert len(response["results"][METRIC]["thresholds"]) == 2
    assert response["abnormalTimeRange"][METRIC] == []


def test_directional_multi_mode_compatibility_is_any_mode_not_average():
    models = [
        ThresholdModel(0, 0, 1, 3, 0.8, 1.2, 0.8, 1.2, 0.1),
        ThresholdModel(1, 2, 3, 3, 4.8, 5.2, 4.8, 5.2, 0.1),
    ]
    assert _classify_value(0, 5.0, models, "DOWN")[0] is False
    assert _classify_value(0, 0.5, models, "DOWN")[0] is True
    assert _classify_value(0, 1.0, models, "BOTH")[0] is False
    assert _classify_value(0, 5.0, models, "BOTH")[0] is False
    assert _classify_value(0, 3.0, models, "BOTH")[0] is True


def test_both_detection_rejects_a_model_with_lower_above_upper():
    invalid = ThresholdModel(0, 0, 1, 3, 2.0, 1.0, 2.0, 1.0, 0.1)
    with pytest.raises(ValueError, match="lower threshold above upper"):
        _classify_value(0, 1.5, [invalid], "BOTH")


def test_multi_metric_failure_is_isolated(tmp_path):
    bad_metric = "broken/metric"
    dataset = make_dataset()
    dataset["standard"][bad_metric] = TimeSeries(
        timestamps=[1, 2, 3],
        values=[1.0],
    )
    dataset["inference"][bad_metric] = TimeSeries(
        timestamps=list(range(100, 106)),
        values=[1.0] * 6,
    )
    response = detector(
        tmp_path,
        dataset,
        metrics=[bad_metric, METRIC],
    ).detect()
    assert bad_metric not in response["states"]
    assert response["metricErrors"][bad_metric] == {
        "code": "metric_input_error",
        "type": "DataValidationError",
        "message": "metric input could not be validated",
    }
    assert "state" not in response["results"][bad_metric]
    assert response["states"][METRIC] == 0


def test_metric_policy_error_is_not_reported_as_business_state(tmp_path):
    bad_metric = "bad-config/metric"
    response = detector(
        tmp_path,
        make_dataset(),
        metrics=[bad_metric, METRIC],
        metric_policies={bad_metric: {"minimum_standard_points": 0}},
    )
    output = response.detect()

    assert bad_metric not in output["states"]
    assert output["metricErrors"][bad_metric]["code"] == "metric_config_error"
    assert output["states"][METRIC] == 0


def test_internal_metric_error_is_serializable_and_redacted(
    tmp_path,
    monkeypatch,
):
    instance = detector(tmp_path, make_dataset())

    def fail_detection(*args, **kwargs):
        raise RuntimeError(
            "token=do-not-expose http://sensitive.internal/full/address"
        )

    monkeypatch.setattr(instance, "build_standard_data", fail_detection)
    response = instance.detect()

    assert METRIC not in response["states"]
    assert response["metricErrors"][METRIC] == {
        "code": "metric_detection_error",
        "type": "RuntimeError",
        "message": "metric detection raised an internal error",
    }
    serialized = json.dumps(response, allow_nan=False)
    assert "do-not-expose" not in serialized
    assert "sensitive.internal" not in serialized


def test_history_is_independent_and_requires_configured_number_of_abnormal_runs(
):
    instance = DegradationPerception(
        dataset=make_dataset(STANDARD_VALUES, UP_INFERENCE),
        metrics=[METRIC],
        task_id="task-a",
        history_policy={"n_keep_result": 3, "n_keep_abnormal": 2},
    )
    first = instance.detect()
    second = instance.detect()
    assert first["abnormalTimeRange"][METRIC] == []
    assert first["results"][METRIC]["currentAbnormalTimeRange"]
    assert second["abnormalTimeRange"][METRIC]
    assert instance.history[("task-a", METRIC)].maxlen == 3


def test_insufficient_run_does_not_advance_valid_history():
    instance = DegradationPerception(
        dataset=make_dataset(STANDARD_VALUES, UP_INFERENCE),
        metrics=[METRIC],
        history_policy={"n_keep_result": 2, "n_keep_abnormal": 2},
    )
    first = instance.detect()
    assert first["abnormalTimeRange"][METRIC] == []
    insufficient = instance.detect_dataset(
        make_dataset(STANDARD_VALUES, [1.5] * 4)
    )
    assert insufficient["states"][METRIC] == 2
    third = instance.detect_dataset(make_dataset(STANDARD_VALUES, UP_INFERENCE))
    assert third["abnormalTimeRange"][METRIC]


def test_detect_dataset_uses_programmatic_timeseries_input():
    instance = DegradationPerception(metrics=[METRIC])
    response: DetectionResult = instance.detect_dataset(make_dataset())
    assert response["states"][METRIC] == 0


def test_cached_standard_model_supports_inference_only_followup(tmp_path):
    instance = detector(
        tmp_path,
        make_dataset(STANDARD_VALUES, NORMAL_INFERENCE),
    )
    assert instance.detect()["states"][METRIC] == 0

    inference_only = {
        "standard": {},
        "inference": make_dataset(
            STANDARD_VALUES,
            UP_INFERENCE,
        )["inference"],
    }
    followup = instance.detect_dataset(inference_only)
    assert followup["states"][METRIC] == 0
    assert followup["abnormalTimeRange"][METRIC]


def test_policy_change_invalidates_cached_standard_model():
    instance = DegradationPerception(
        dataset=make_dataset(),
        metrics=[METRIC],
    )
    assert instance.detect()["states"][METRIC] == 0

    instance.metric_policies[METRIC] = {"upper_ratio": 1.20}
    inference_only = {
        "standard": {},
        "inference": make_dataset()["inference"],
    }
    response = instance.detect_dataset(inference_only)
    assert response["states"][METRIC] == 1
    assert response["results"][METRIC]["thresholds"] == []
