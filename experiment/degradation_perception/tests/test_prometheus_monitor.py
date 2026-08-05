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

from experiment.degradation_perception.baseline_model import (
    fit_baseline_model,
    load_baseline_model,
)
from experiment.degradation_perception.perception_config import TimeSeries
from experiment.degradation_perception.prometheus_monitor import (
    BaselineBootstrapConfig,
    PrometheusConnectionConfig,
    PrometheusMonitor,
    PrometheusMonitorConfig,
    TargetSelectionConfig,
    load_prometheus_monitor_config,
)
from experiment.degradation_perception.prometheus_source import (
    PrometheusMetricQuery,
)

TARGET = "timing_s/step"
CANDIDATE = "actor/entropy"
FIXED_NOW = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)


def _write_policy(path: Path) -> None:
    path.write_text("{}\n", encoding="utf-8")


def test_config_uses_time_timing_defaults_and_explicit_overrides(tmp_path: Path):
    policy = tmp_path / "algorithm.yaml"
    _write_policy(policy)
    config = tmp_path / "prometheus.yaml"
    config.write_text(
        "baseline_model: ./baseline.json\n"
        "policy_config: ./algorithm.yaml\n"
        "prometheus:\n"
        "  url: http://prometheus:9090\n"
        "  global_step_query: training_global_step\n"
        "metrics:\n"
        "  timing_s/step:\n"
        "    promql: timing_metric\n"
        "  actor/entropy:\n"
        "    promql: entropy_metric\n"
        "  custom_metric:\n"
        "    promql: custom_metric\n"
        "target_selection:\n"
        "  patterns: [time, timing]\n"
        "  overrides:\n"
        "    custom_metric: target\n",
        encoding="utf-8",
    )

    loaded = load_prometheus_monitor_config(config)

    assert loaded.target_metrics == (TARGET, "custom_metric")
    assert loaded.candidate_metrics == (CANDIDATE,)
    assert loaded.poll_interval_seconds == 180
    assert loaded.prometheus.query_step_seconds == 15


class FakePrometheusClient:
    def __init__(self, global_steps: TimeSeries, metrics: dict[str, TimeSeries]):
        self.global_steps = global_steps
        self.metrics = metrics
        self.query_calls = []
        self.fetch_calls = []

    def query_range(self, query, *, start, end, step):
        self.query_calls.append((query, start, end, step))
        return self.global_steps

    def fetch_range(self, queries, *, start, end, step):
        self.fetch_calls.append(
            (tuple(query.name for query in queries), start, end, step)
        )
        return {
            query.name: self.metrics.get(query.name, TimeSeries()) for query in queries
        }


class FakeRunner:
    def __init__(self):
        self.calls = 0
        self.restore_calls = 0
        self.abnormal = True

    def minimum_inference_samples(self):
        return {TARGET: 2, CANDIDATE: 2}

    def snapshot_runtime_state(self):
        return {"calls": self.calls}

    def restore_runtime_state(self, snapshot):
        self.restore_calls += 1
        self.calls = snapshot["calls"]

    def run(self, inference):
        self.calls += 1
        return {
            "taskId": "prometheus-test",
            "states": {TARGET: 0, CANDIDATE: 0},
            "results": {
                TARGET: {"currentAbnormalTimeRange": []},
                CANDIDATE: {"currentAbnormalTimeRange": []},
            },
            "abnormalTimeRange": {
                TARGET: (
                    [
                        {
                            "startTime": "2026-08-05T11:00:00+00:00",
                            "endTime": "2026-08-05T11:03:00+00:00",
                            "abnormalType": "UP",
                        }
                    ]
                    if self.abnormal
                    else []
                ),
                CANDIDATE: [],
            },
            "associationAnalysis": {
                "targets": {
                    TARGET: {
                        "events": [
                            {
                                "topAssociations": [
                                    {"rank": 1, "metric": CANDIDATE, "score": 0.9}
                                ]
                            }
                        ]
                    }
                }
            },
        }


def _baseline():
    timestamps = list(range(1, 31))
    target_values = [1.0 + (index % 5 - 2) * 0.002 for index in timestamps]
    candidate_values = [5.0 + (index % 5 - 2) * 0.01 for index in timestamps]
    return fit_baseline_model(
        {
            TARGET: TimeSeries(timestamps, target_values),
            CANDIDATE: TimeSeries(timestamps, candidate_values),
        },
        [TARGET, CANDIDATE],
        source_type="prometheus",
    )


def _runtime_config(tmp_path: Path, policy: Path) -> PrometheusMonitorConfig:
    return PrometheusMonitorConfig(
        baseline_model=tmp_path / "baseline.json",
        policy_config=policy,
        prometheus=PrometheusConnectionConfig(
            url="http://prometheus:9090",
            global_step_query="training_global_step",
            query_step_seconds=15,
            lookback_seconds=3600,
            overlap_seconds=30,
            alignment_tolerance_seconds=8,
        ),
        queries=(
            PrometheusMetricQuery(TARGET, "timing_metric", "target", "UP"),
            PrometheusMetricQuery(CANDIDATE, "entropy_metric", "candidate", "BOTH"),
        ),
        target_metrics=(TARGET,),
        candidate_metrics=(CANDIDATE,),
        baseline=BaselineBootstrapConfig(),
        target_selection=TargetSelectionConfig(),
        poll_interval_seconds=180,
        inference_window_points=100,
        state_file=tmp_path / "output" / "state.json",
        output_dir=tmp_path / "output",
        task_id="prometheus-test",
    )


def test_repeated_abnormal_poll_writes_only_one_event_file(tmp_path: Path):
    policy = tmp_path / "algorithm.yaml"
    _write_policy(policy)
    timestamps = [1000.0, 1015.0, 1030.0]
    client = FakePrometheusClient(
        TimeSeries(timestamps, [26, 27, 28]),
        {
            TARGET: TimeSeries(timestamps, [2.0, 2.1, 2.2]),
            CANDIDATE: TimeSeries(timestamps, [7.0, 7.1, 7.2]),
        },
    )
    runner = FakeRunner()
    monitor = PrometheusMonitor(
        _runtime_config(tmp_path, policy),
        client=client,
        baseline=_baseline(),
        runner=runner,
        now=lambda: FIXED_NOW,
    )

    first = monitor.poll_once()
    next_timestamps = timestamps + [1045.0]
    client.global_steps = TimeSeries(next_timestamps, [26, 27, 28, 29])
    client.metrics = {
        TARGET: TimeSeries(next_timestamps, [2.0, 2.1, 2.2, 2.3]),
        CANDIDATE: TimeSeries(next_timestamps, [7.0, 7.1, 7.2, 7.3]),
    }
    second = monitor.poll_once()

    assert first.status == "state_changed"
    assert first.event_path is not None and first.event_path.is_file()
    assert first.result is not None
    assert first.result["events"][0]["targetMetric"] == TARGET
    assert first.result["events"][0]["topAssociations"][0]["metric"] == CANDIDATE
    assert second.status == "detected"
    assert second.event_path is None
    assert second.result is not None and second.result["events"] == []
    assert len(list((tmp_path / "output").glob("event_*.json"))) == 1
    state = json.loads((tmp_path / "output" / "state.json").read_text())
    assert set(state["activeEventIds"]) == {TARGET}


def test_unchanged_global_step_does_not_rerun_detector(tmp_path: Path):
    policy = tmp_path / "algorithm.yaml"
    _write_policy(policy)
    timestamps = [1000.0, 1015.0, 1030.0]
    client = FakePrometheusClient(
        TimeSeries(timestamps, [26, 27, 28]),
        {
            TARGET: TimeSeries(timestamps, [2.0, 2.1, 2.2]),
            CANDIDATE: TimeSeries(timestamps, [7.0, 7.1, 7.2]),
        },
    )
    runner = FakeRunner()
    monitor = PrometheusMonitor(
        _runtime_config(tmp_path, policy),
        client=client,
        baseline=_baseline(),
        runner=runner,
        now=lambda: FIXED_NOW,
    )

    first = monitor.poll_once()
    second = monitor.poll_once()

    assert first.status == "state_changed"
    assert second.status == "no_new_data"
    assert runner.calls == 1


def test_normal_window_after_active_fault_emits_recovery(tmp_path: Path):
    policy = tmp_path / "algorithm.yaml"
    _write_policy(policy)
    timestamps = [1000.0, 1015.0, 1030.0]
    client = FakePrometheusClient(
        TimeSeries(timestamps, [26, 27, 28]),
        {
            TARGET: TimeSeries(timestamps, [2.0, 2.1, 2.2]),
            CANDIDATE: TimeSeries(timestamps, [7.0, 7.1, 7.2]),
        },
    )
    runner = FakeRunner()
    monitor = PrometheusMonitor(
        _runtime_config(tmp_path, policy),
        client=client,
        baseline=_baseline(),
        runner=runner,
        now=lambda: FIXED_NOW,
    )
    abnormal = monitor.poll_once()
    assert abnormal.result is not None
    active_event_id = abnormal.result["events"][0]["eventId"]

    runner.abnormal = False
    next_timestamps = timestamps + [1045.0]
    client.global_steps = TimeSeries(next_timestamps, [26, 27, 28, 29])
    client.metrics = {
        TARGET: TimeSeries(next_timestamps, [2.0, 2.1, 2.2, 1.0]),
        CANDIDATE: TimeSeries(next_timestamps, [7.0, 7.1, 7.2, 5.0]),
    }
    recovered = monitor.poll_once()

    assert recovered.status == "state_changed"
    assert recovered.result is not None
    assert recovered.result["events"] == [
        {
            "eventId": recovered.result["events"][0]["eventId"],
            "eventType": "RECOVERED",
            "detectedAt": FIXED_NOW.isoformat(),
            "targetMetric": TARGET,
            "activeEventId": active_event_id,
            "topAssociations": [],
        }
    ]
    assert monitor.state is not None and monitor.state.active_event_ids == {}


def test_missing_baseline_bootstraps_steps_5_to_25_then_detects(tmp_path: Path):
    policy = tmp_path / "algorithm.yaml"
    _write_policy(policy)
    config = _runtime_config(tmp_path, policy)
    steps = list(range(5, 36))
    timestamps = [1000.0 + step * 15 for step in steps]
    target_values = [1.0 + (step % 5 - 2) * 0.002 for step in steps]
    candidate_values = [5.0 + (step % 5 - 2) * 0.01 for step in steps]
    client = FakePrometheusClient(
        TimeSeries(timestamps, steps),
        {
            TARGET: TimeSeries(timestamps, target_values),
            CANDIDATE: TimeSeries(timestamps, candidate_values),
        },
    )
    monitor = PrometheusMonitor(
        config,
        client=client,
        now=lambda: FIXED_NOW,
    )

    outcome = monitor.poll_once()

    assert outcome.status in {"detected", "state_changed"}
    assert config.baseline_model.is_file()
    fitted = load_baseline_model(config.baseline_model)
    assert fitted.source_type == "prometheus"
    assert fitted.metrics[TARGET].standard_samples == 21
    assert fitted.metrics[CANDIDATE].standard_samples == 21
    assert monitor.state is not None
    assert monitor.state.last_global_step == 35
    assert len(client.query_calls) == 2
