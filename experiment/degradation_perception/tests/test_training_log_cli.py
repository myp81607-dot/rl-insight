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

from experiment.degradation_perception import main as cli


def _write_log(path, rows):
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _minimal_result(metric: str, task_id: str) -> dict:
    return {
        "taskId": task_id,
        "states": {metric: 0},
        "results": {metric: {}},
        "abnormalTimeRange": {metric: []},
        "debugDetails": {
            "metrics": {
                metric: {
                    "state": 0,
                    "standardSamples": 2,
                    "inferenceSamples": 2,
                    "alpha": 0.01,
                    "lowerProbability": 0.01,
                    "upperProbability": 0.99,
                    "normalModelCount": 1,
                    "normalModels": [],
                    "abnormalPointCount": 0,
                    "abnormalIntervalCount": 0,
                }
            }
        },
    }


def test_list_metrics_prints_only_first_seen_original_names(tmp_path, monkeypatch, capsys):
    standard = tmp_path / "healthy.log"
    _write_log(
        standard,
        [
            "step:0 - global_seqlen/min:1 - actor/entropy:2",
            "step:1 - actor/entropy:3 - timing_s/step:4",
        ],
    )

    class ForbiddenRunner:
        def __init__(self, **_kwargs):
            raise AssertionError("--list-metrics must not run detection")

    monkeypatch.setattr(cli, "DetectionRunner", ForbiddenRunner)
    assert cli.main(["--standard-log", str(standard), "--list-metrics"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "global_seqlen/min",
        "actor/entropy",
        "timing_s/step",
    ]


def test_cli_keeps_healthy_and_inference_phases_separate(tmp_path, monkeypatch):
    metric = "timing_s/step"
    standard = tmp_path / "healthy.log"
    inference = tmp_path / "inference.log"
    output = tmp_path / "result.json"
    _write_log(standard, [f"step:0 - {metric}:1", f"step:1 - {metric}:2"])
    _write_log(inference, [f"step:0 - {metric}:10", f"step:1 - {metric}:20"])
    captured = {}

    class RecordingRunner:
        def __init__(self, *, standard, metrics, task_id, **_kwargs):
            captured["standard"] = standard
            captured["metrics"] = metrics
            captured["task_id"] = task_id

        def run(self, current):
            captured["inference"] = current
            return _minimal_result(metric, captured["task_id"])

    monkeypatch.setattr(cli, "DetectionRunner", RecordingRunner)
    assert cli.main(
        [
            "--standard-log",
            str(standard),
            "--inference-log",
            str(inference),
            "--metrics",
            metric,
            "--task-id",
            "phase-test",
            "--output",
            str(output),
        ]
    ) == 0
    assert captured["standard"][metric].timestamps == [0.0, 1.0]
    assert captured["standard"][metric].values == [1.0, 2.0]
    assert captured["inference"][metric].timestamps == [0.0, 1.0]
    assert captured["inference"][metric].values == [10.0, 20.0]


def test_real_cli_completes_when_inference_has_no_anomaly(tmp_path):
    metric = "timing_s/step"
    standard = tmp_path / "healthy.log"
    inference = tmp_path / "inference.log"
    output = tmp_path / "result.json"
    healthy_values = [1.00, 1.01, 0.99, 1.02, 1.00, 0.98] * 5
    normal_values = [1.00, 1.01, 0.99, 1.00, 1.02, 0.98]
    _write_log(
        standard,
        [f"step:{index} - {metric}:{value}" for index, value in enumerate(healthy_values)],
    )
    _write_log(
        inference,
        [f"step:{index} - {metric}:{value}" for index, value in enumerate(normal_values)],
    )

    assert cli.main(
        [
            "--standard-log",
            str(standard),
            "--inference-log",
            str(inference),
            "--metrics",
            metric,
            "--output",
            str(output),
        ]
    ) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["states"] == {metric: 0}
    assert result["abnormalTimeRange"] == {metric: []}
    assert result["debugDetails"]["metrics"][metric]["normalModels"]


def test_debug_terminal_rounds_only_display_and_json_keeps_precision(
    tmp_path, monkeypatch, capsys
):
    metric = "timing_s/step"
    standard = tmp_path / "healthy.log"
    inference = tmp_path / "inference.log"
    output = tmp_path / "result.json"
    _write_log(standard, [f"step:0 - {metric}:1"])
    _write_log(inference, [f"step:0 - {metric}:2"])
    result = _minimal_result(metric, "precision-test")
    model = {
        "rawLowerThreshold": 1.234567890123,
        "rawUpperThreshold": 9.876543210987,
        "finalLowerThreshold": 0.123456789012,
        "finalUpperThreshold": 9.876543210987,
        "lowerRatio": 1.15,
        "upperRatio": 1.15,
        "effectiveBandwidth": 0.123456789012,
    }
    result["debugDetails"]["metrics"][metric]["normalModels"] = [model]

    class DebugRunner:
        def __init__(self, **_kwargs):
            pass

        def run(self, _inference):
            return result

    monkeypatch.setattr(cli, "DetectionRunner", DebugRunner)
    assert cli.main(
        [
            "--standard-log",
            str(standard),
            "--inference-log",
            str(inference),
            "--metrics",
            metric,
            "--debug-kde",
            "--output",
            str(output),
        ]
    ) == 0
    terminal = capsys.readouterr().out
    assert "Raw KDE range: 1.23 ~ 9.88" in terminal
    assert "KDE standard range: 0.12 ~ 9.88" in terminal
    assert "KDE bandwidth: 0.12" in terminal
    saved = json.loads(output.read_text(encoding="utf-8"))
    saved_model = saved["debugDetails"]["metrics"][metric]["normalModels"][0]
    assert saved_model["rawLowerThreshold"] == 1.234567890123
    assert saved_model["effectiveBandwidth"] == 0.123456789012
