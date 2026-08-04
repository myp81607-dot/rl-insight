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
from pathlib import Path

import pytest

import experiment.degradation_perception.algorithm as algorithm_module
from experiment.degradation_perception import main as cli
from experiment.degradation_perception.baseline_model import load_baseline_model


METRIC = "timing_s/step"
HEALTHY = [1.00, 1.01, 0.99, 1.02, 1.00, 0.98, 1.01, 1.00, 0.99, 1.02] * 3
INFERENCE = [1.00, 1.01, 0.99, 1.00, 1.02, 0.98]


def _write_log(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"step:{index} - {METRIC}:{value}\n"
            for index, value in enumerate(values)
        ),
        encoding="utf-8",
    )


def _fit(tmp_path: Path) -> tuple[Path, Path]:
    healthy = tmp_path / "data" / "healthy.log"
    model = tmp_path / "models" / "baseline_model.json"
    _write_log(healthy, HEALTHY)
    assert cli.main(
        [
            "fit",
            "--standard-log",
            str(healthy),
            "--metrics",
            METRIC,
            "--output-model",
            str(model),
        ]
    ) == 0
    return healthy, model


def test_fit_then_detect_uses_saved_model_without_kde_refit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _healthy, model = _fit(tmp_path)
    inference = tmp_path / "data" / "inference.log"
    config = tmp_path / "real_test.json"
    output = tmp_path / "new" / "nested" / "result.json"
    _write_log(inference, INFERENCE)
    config.write_text(
        json.dumps(
            {
                "baseline_model": "./models/baseline_model.json",
                "inference_log": "./data/inference.log",
                "metrics": [METRIC],
                "task_id": "real-test-001",
                "debug_kde": True,
                "output": "./new/nested/result.json",
            }
        ),
        encoding="utf-8",
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("detect must not fit KDE")

    monkeypatch.setattr(algorithm_module.DegradationPerception, "build_standard_data", forbidden)

    assert cli.main(["detect", "--config", str(config)]) == 0
    assert output.is_file()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["states"] == {METRIC: 0}
    assert set(result) >= {
        "states",
        "abnormalTimeRange",
        "results",
        "debugDetails",
    }
    saved_model = load_baseline_model(model)
    assert result["debugDetails"]["metrics"][METRIC]["standardSamples"] == saved_model.metrics[METRIC].standard_samples


def test_detect_reads_only_the_inference_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    healthy, model = _fit(tmp_path)
    inference = tmp_path / "data" / "fault.log"
    _write_log(inference, [10.0] * 6)
    output = tmp_path / "result.json"
    config = tmp_path / "real_test.json"
    config.write_text(
        json.dumps(
            {
                "baseline_model": str(model),
                "inference_log": str(inference),
                "metrics": [METRIC],
                "task_id": "read-isolation",
                "debug_kde": False,
                "output": str(output),
            }
        ),
        encoding="utf-8",
    )
    real_loader = cli.load_verl_training_log
    loaded_paths: list[Path] = []

    def recording_loader(path):
        loaded_paths.append(Path(path))
        return real_loader(path)

    monkeypatch.setattr(cli, "load_verl_training_log", recording_loader)

    assert cli.main(["detect", "--config", str(config)]) == 0
    assert loaded_paths == [inference]
    assert healthy not in loaded_paths
    result = json.loads(output.read_text(encoding="utf-8"))
    published = result["abnormalTimeRange"][METRIC][0]
    assert result["abnormalMetrics"] == [
        {
            "metric": METRIC,
            "events": [
                {
                    "eventIndex": 0,
                    "startTime": published["startTime"],
                    "endTime": published["endTime"],
                }
            ],
        }
    ]


def test_real_test_config_rejects_unknown_healthy_field(tmp_path: Path):
    config = tmp_path / "real_test.json"
    config.write_text(
        json.dumps(
            {
                "baseline_model": "baseline.json",
                "inference_log": "inference.log",
                "metrics": [METRIC],
                "task_id": "invalid",
                "debug_kde": False,
                "output": "result.json",
                "healthy_log": "healthy.log",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported fields"):
        cli.load_real_test_config(config)


def test_fit_reads_algorithm_parameters_from_yaml(tmp_path: Path):
    healthy = tmp_path / "healthy.log"
    model = tmp_path / "baseline.json"
    policy = tmp_path / "algorithm.yaml"
    _write_log(healthy, HEALTHY)
    policy.write_text(
        "alpha: 0.2\n"
        "upper_ratio: 1.25\n"
        "history_policy:\n"
        "  n_keep_result: 3\n"
        "  n_keep_abnormal: 2\n",
        encoding="utf-8",
    )

    assert cli.main(
        [
            "fit",
            "--standard-log",
            str(healthy),
            "--metrics",
            METRIC,
            "--output-model",
            str(model),
            "--policy-config",
            str(policy),
        ]
    ) == 0

    loaded = load_baseline_model(model)
    assert loaded.metrics[METRIC].policy["alpha"] == pytest.approx(0.2)
    assert loaded.metrics[METRIC].policy["upper_ratio"] == pytest.approx(1.25)
    assert loaded.history_policy == {
        "n_keep_result": 3,
        "n_keep_abnormal": 2,
    }
