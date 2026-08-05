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
from datetime import datetime, timezone
from pathlib import Path

import pytest

from experiment.degradation_perception.perception_config import TimeSeries
from experiment.degradation_perception.prometheus_monitor import (
    MonitorConfig,
    PrometheusRuntime,
    load_config,
    main,
    select_analysis,
)
from experiment.degradation_perception.prometheus_source import PrometheusSeries


FIXED_NOW = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)


class FakePrometheusClient:
    def __init__(
        self,
        series: list[PrometheusSeries],
        values: dict[str, TimeSeries],
    ) -> None:
        self.series = series
        self.values = values
        self.list_calls = []
        self.query_calls = []

    def list_series(self, match, *, start, end):
        self.list_calls.append((match, start, end))
        return list(self.series)

    def query_range(self, query, *, start, end, step):
        self.query_calls.append((query, start, end, step))
        return self.values.get(query, TimeSeries())


def _config(tmp_path: Path) -> MonitorConfig:
    package = Path(__file__).resolve().parents[1]
    return MonitorConfig(
        prometheus_url="http://prometheus:9090",
        series_selector='{__name__=~".+"}',
        query_step_seconds=1,
        lookback_seconds=3600,
        alignment_tolerance_seconds=0.5,
        timeout_seconds=30,
        verify_tls=True,
        baseline_start_step=5,
        baseline_end_step=25,
        minimum_baseline_samples=10,
        inference_points=120,
        poll_interval_seconds=180,
        top_k=5,
        target_patterns=("time", "timing"),
        global_step_patterns=("global_step",),
        algorithm_config=package / "algorithm_config.yaml",
        output_dir=tmp_path / "output",
    )


def _series_data():
    labels = {"experiment": "smoke", "job": "trainer"}
    global_step = PrometheusSeries("training_global_step", labels)
    target = PrometheusSeries("rl_insight_timing_s_step", labels)
    strong = PrometheusSeries("actor_entropy", labels)
    normal = PrometheusSeries("critic_loss", labels)
    short = PrometheusSeries("short_candidate", labels)

    steps = list(range(1, 61))
    timestamps = [1000.0 + step for step in steps]
    healthy_wave = [((step % 5) - 2) for step in steps]
    values = {
        global_step.selector: TimeSeries(timestamps, steps),
        target.selector: TimeSeries(
            timestamps,
            [
                1.0 + wave * 0.002 if step <= 25 else 1.8 + wave * 0.002
                for step, wave in zip(steps, healthy_wave)
            ],
        ),
        strong.selector: TimeSeries(
            timestamps,
            [
                10.0 + wave * 0.02 if step <= 25 else 18.0 + wave * 0.02
                for step, wave in zip(steps, healthy_wave)
            ],
        ),
        normal.selector: TimeSeries(
            timestamps,
            [5.0 + wave * 0.01 for wave in healthy_wave],
        ),
        short.selector: TimeSeries(timestamps[:3], [1.0, 1.1, 1.2]),
    }
    return [global_step, target, strong, normal, short], values


def test_config_defaults_to_all_series_and_step_5_to_25(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "prometheus_url: http://prometheus:9090\n"
        "algorithm_config: algorithm.yaml\n",
        encoding="utf-8",
    )
    (tmp_path / "algorithm.yaml").write_text("{}\n", encoding="utf-8")

    config = load_config(config_path)

    assert config.series_selector == '{__name__=~".+"}'
    assert (config.baseline_start_step, config.baseline_end_step) == (5, 25)
    assert config.target_patterns == ("time", "timing")
    assert config.poll_interval_seconds == 180


def test_query_train_detect_and_analyze_end_to_end(tmp_path: Path):
    series, values = _series_data()
    client = FakePrometheusClient(series, values)
    config = _config(tmp_path)
    runtime = PrometheusRuntime(config, client=client, now=lambda: FIXED_NOW)

    queried = runtime.query()
    assert queried["summary"] == {
        "series": 5,
        "globalSteps": 1,
        "targets": 1,
        "candidates": 3,
        "failedQueries": 0,
    }
    assert config.query_file.is_file()
    assert queried["series"][1]["samples"]["values"]

    trained = runtime.train()
    assert trained["trainingInterval"]["startStep"] == 5
    assert trained["trainingInterval"]["endStep"] == 25
    assert trained["summary"] == {
        "discovered": 5,
        "trained": 3,
        "targets": 1,
        "candidates": 2,
        "skipped": 1,
    }
    assert trained["skippedSeries"][0]["metricName"] == "short_candidate"
    assert config.training_file.is_file()

    detected = runtime.detect()
    assert detected["status"] == "abnormal"
    assert detected["lastGlobalStep"] == 60
    assert detected["summary"]["abnormalTargets"] == 1
    target = detected["targets"][0]
    assert target["metricName"] == "rl_insight_timing_s_step"
    assert target["top5"][0]["metricName"] == "actor_entropy"
    assert target["top5"][0]["abnormalContributionPercent"] == pytest.approx(100)
    json.loads(config.result_file.read_text(encoding="utf-8"))

    view = select_analysis(detected, "timing_s_step")
    assert view["targets"][0]["top5"][0]["metricName"] == "actor_entropy"


def test_analyze_supports_metric_name_as_flag(tmp_path: Path, capsys):
    config_path = tmp_path / "config.yaml"
    algorithm = Path(__file__).resolve().parents[1] / "algorithm_config.yaml"
    config_path.write_text(
        f"algorithm_config: {algorithm}\noutput_dir: ./output\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "anomalies.json").write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "id": "rl_timing_s_step{job=trainer}",
                        "metricName": "rl_timing_s_step",
                        "labels": {"job": "trainer"},
                        "status": "abnormal",
                        "intervals": [{"startTime": 1, "endTime": 2}],
                        "top5": [
                            {
                                "rank": 1,
                                "metricName": "actor_entropy",
                                "abnormalContributionPercent": 100,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    status = main(["analyze", "--config", str(config_path), "--time_step"])

    assert status == 0
    assert '"actor_entropy"' in capsys.readouterr().out
