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

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

import experiment.degradation_perception.algorithm as algorithm_module
import experiment.degradation_perception.baseline_model as baseline_module
from experiment.degradation_perception.baseline_model import (
    BaselineDetectionRunner,
    MetricBaseline,
    fit_baseline_model,
    load_baseline_model,
    save_baseline_model,
)
from experiment.degradation_perception.perception_config import TimeSeries


METRIC = "timing_s/step"
HEALTHY = [1.00, 1.01, 0.99, 1.02, 1.00, 0.98, 1.01, 1.00, 0.99, 1.02] * 3


def _series(values, start=0) -> TimeSeries:
    return TimeSeries(
        timestamps=list(range(start, start + len(values))),
        values=list(values),
    )


def _fitted():
    return fit_baseline_model(
        {METRIC: _series(HEALTHY)},
        [METRIC],
        created_at="2026-08-04T00:00:00+00:00",
    )


def test_fit_saves_and_loads_complete_baseline_model(tmp_path: Path):
    model = _fitted()
    output = tmp_path / "models" / "baseline_model.json"

    save_baseline_model(model, output)
    loaded = load_baseline_model(output, expected_metrics=[METRIC])
    raw = json.loads(output.read_text(encoding="utf-8"))

    assert loaded.to_dict() == model.to_dict()
    assert raw["schemaVersion"] == 1
    assert raw["baselineId"] == "default"
    assert raw["sourceType"] == "training_log"
    assert list(raw["metrics"]) == [METRIC]
    metric = raw["metrics"][METRIC]
    assert metric["normalModels"]
    assert metric["standardSamples"] == len(HEALTHY)
    assert "inference" not in raw
    assert "inferenceData" not in metric
    assert "abnormalTimeRange" not in raw
    for normal_model in metric["normalModels"]:
        assert normal_model["rawLowerThreshold"] < normal_model["rawUpperThreshold"]
        assert normal_model["finalLowerThreshold"] < normal_model["finalUpperThreshold"]
        assert normal_model["bandwidth"] > 0


def test_multiple_normal_modes_remain_separate_after_round_trip(tmp_path: Path):
    fitted = _fitted()
    metric = fitted.metrics[METRIC]
    first = metric.models[0]
    second = replace(
        first,
        mode_id=first.mode_id + 100,
        segment_start_time=100.0,
        segment_end_time=129.0,
        lower_kde_threshold=first.lower_kde_threshold + 10.0,
        upper_kde_threshold=first.upper_kde_threshold + 10.0,
        lower_threshold=first.lower_threshold + 10.0,
        upper_threshold=first.upper_threshold + 10.0,
    )
    multi = replace(
        fitted,
        metrics={
            METRIC: MetricBaseline(
                metric=METRIC,
                policy=copy.deepcopy(metric.policy),
                standard_samples=metric.standard_samples,
                models=(first, second),
            )
        },
    )
    output = tmp_path / "baseline_model.json"

    save_baseline_model(multi, output)
    loaded = load_baseline_model(output)

    assert len(loaded.metrics[METRIC].models) == 2
    assert [item.mode_id for item in loaded.metrics[METRIC].models] == [
        first.mode_id,
        second.mode_id,
    ]
    assert loaded.metrics[METRIC].models[0].upper_threshold != loaded.metrics[METRIC].models[1].upper_threshold


def test_loaded_runner_never_refits_kde(monkeypatch: pytest.MonkeyPatch):
    baseline = _fitted()
    runner = BaselineDetectionRunner(baseline, metrics=[METRIC])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("detect must not build standard data")

    monkeypatch.setattr(algorithm_module.DegradationPerception, "build_standard_data", forbidden)
    result = runner.run({METRIC: _series([1.0, 1.01, 0.99, 1.0, 1.02, 0.98], 100)})

    assert result["states"] == {METRIC: 0}
    assert result["debugDetails"]["metrics"][METRIC]["standardSamples"] == len(HEALTHY)


def test_changing_inference_does_not_change_loaded_thresholds():
    baseline = _fitted()
    runner = BaselineDetectionRunner(baseline, metrics=[METRIC])
    before = copy.deepcopy(runner.detector.standard_models)

    runner.run({METRIC: _series([1.0] * 6, 100)})
    runner.run({METRIC: _series([10.0] * 6, 200)})

    assert runner.detector.standard_models == before
    assert baseline.to_dict() == runner.baseline.to_dict()


def test_loaded_runner_accepts_runtime_policy_overrides():
    runner = BaselineDetectionRunner(
        _fitted(),
        metrics=[METRIC],
        metric_policies={
            METRIC: {
                "abnormal_type": "DOWN",
                "abnormal_interval": {"minimum_abnormal_points": 2},
            }
        },
        history_policy={"n_keep_result": 3, "n_keep_abnormal": 2},
    )

    policy = runner.metric_policies[METRIC]
    assert policy["abnormal_type"] == "DOWN"
    assert policy["abnormal_interval"]["minimum_abnormal_points"] == 2
    assert runner.detector.n_keep_result == 3
    assert runner.detector.n_keep_abnormal == 2


def test_loaded_runner_rejects_yaml_change_that_requires_refit():
    with pytest.raises(ValueError, match="alpha.*run fit again"):
        BaselineDetectionRunner(
            _fitted(),
            metrics=[METRIC],
            metric_policies={METRIC: {"alpha": 0.2}},
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda raw: raw.update(schemaVersion=99), "schemaVersion"),
        (
            lambda raw: raw["metrics"][METRIC].update(normalModels=[]),
            "normalModels",
        ),
        (
            lambda raw: raw["metrics"][METRIC]["normalModels"][0].update(
                finalLowerThreshold=100.0,
                finalUpperThreshold=10.0,
            ),
            "final lower threshold",
        ),
    ],
)
def test_invalid_baseline_files_fail_explicitly(tmp_path: Path, mutation, match: str):
    raw = _fitted().to_dict()
    mutation(raw)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_baseline_model(path)


def test_requested_metric_must_exist_in_baseline(tmp_path: Path):
    path = tmp_path / "baseline.json"
    save_baseline_model(_fitted(), path)

    with pytest.raises(ValueError, match="absent from baseline"):
        load_baseline_model(path, expected_metrics=["actor/entropy"])


def test_successful_refit_replaces_model_and_backs_up_old_file(tmp_path: Path):
    output = tmp_path / "models" / "baseline_model.json"
    first = _fitted()
    second = replace(first, baseline_id="second")
    save_baseline_model(first, output)
    old_bytes = output.read_bytes()

    save_baseline_model(second, output)

    assert load_baseline_model(output).baseline_id == "second"
    backups = list((output.parent / "history").glob("baseline_model_*.json"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == old_bytes


def test_failed_refit_or_replace_does_not_damage_old_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "models" / "baseline_model.json"
    old = _fitted()
    save_baseline_model(old, output)
    old_bytes = output.read_bytes()

    with pytest.raises(ValueError, match="fewer than"):
        fit_baseline_model({METRIC: _series([1.0, 1.0])}, [METRIC])
    assert output.read_bytes() == old_bytes

    real_replace = baseline_module.os.replace

    def fail_destination(source, destination):
        if Path(destination) == output:
            raise OSError("simulated atomic replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(baseline_module.os, "replace", fail_destination)
    with pytest.raises(OSError, match="simulated"):
        save_baseline_model(replace(old, baseline_id="new"), output)
    assert output.read_bytes() == old_bytes
