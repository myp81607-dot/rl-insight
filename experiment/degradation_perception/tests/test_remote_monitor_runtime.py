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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

import experiment.degradation_perception.remote_monitor as remote_monitor_module
from experiment.degradation_perception.perception_config import TimeSeries
from experiment.degradation_perception.remote_monitor import (
    RemoteConnectionConfig,
    RemoteConnectionError,
    RemoteMonitor,
    RemoteMonitorConfig,
    load_remote_monitor_config,
)


METRIC = "timing_s/step"
FIXED_NOW = datetime(2026, 8, 3, 12, 30, tzinfo=timezone.utc)


class FakeReader:
    def __init__(self, *events: str | BaseException) -> None:
        self.events = list(events)
        self.calls = 0

    def read_snapshot(self) -> str:
        if self.calls >= len(self.events):
            raise AssertionError("fake reader has no event for this poll")
        event = self.events[self.calls]
        self.calls += 1
        if isinstance(event, BaseException):
            raise event
        return event


class FakeRunner:
    def __init__(self, minimum_samples: int) -> None:
        self.minimum_samples = minimum_samples
        self.calls: list[dict[str, TimeSeries]] = []

    def minimum_inference_samples(self) -> dict[str, int]:
        return {METRIC: self.minimum_samples}

    def run(self, inference: dict[str, TimeSeries]) -> dict[str, Any]:
        self.calls.append(
            {
                metric: TimeSeries(list(series.timestamps), list(series.values))
                for metric, series in inference.items()
            }
        )
        return {
            "taskId": "remote-runtime-test",
            "states": {METRIC: 0},
            "results": {METRIC: {"currentAbnormalTimeRange": []}},
            "abnormalTimeRange": {METRIC: []},
            "debugDetails": {
                "metrics": {
                    METRIC: {
                        "alpha": 0.0123456789,
                        "normalModels": [
                            {
                                "rawLowerThreshold": 1.23456789,
                                "rawUpperThreshold": 9.87654321,
                            }
                        ],
                    }
                }
            },
        }


def _make_monitor(
    tmp_path: Path,
    *,
    reader: FakeReader,
    runner: FakeRunner,
) -> RemoteMonitor:
    standard_log = tmp_path / "healthy.log"
    standard_log.write_text(
        "step:0 - timing_s/step:10.0\n"
        "step:1 - timing_s/step:11.0\n"
        "step:2 - timing_s/step:12.0\n",
        encoding="utf-8",
    )
    config = RemoteMonitorConfig(
        standard_log=standard_log,
        remote=RemoteConnectionConfig(
            host="private-training-host.internal",
            username="monitor-user",
            log_path="/srv/private/trainer.log",
            password_env="REMOTE_MONITOR_SECRET",
        ),
        metrics=(METRIC,),
        poll_interval_seconds=30,
        output_dir=tmp_path / "monitor-output",
        task_id="remote-runtime-test",
    )
    return RemoteMonitor(
        config,
        reader=reader,
        runner=runner,  # type: ignore[arg-type]
        now=lambda: FIXED_NOW,
    )


def test_only_steps_strictly_above_last_step_reach_detector(tmp_path: Path):
    reader = FakeReader(
        "step:4 - timing_s/step:4.0\n"
        "step:5 - timing_s/step:5.0\n"
        "step:6 - timing_s/step:6.0\n",
        "step:4 - timing_s/step:4.0\n"
        "step:5 - timing_s/step:5.0\n"
        "step:6 - timing_s/step:6.0\n",
    )
    runner = FakeRunner(minimum_samples=1)
    monitor = _make_monitor(tmp_path, reader=reader, runner=runner)
    monitor.state.last_step = 5

    detected = monitor.poll_once()
    no_new = monitor.poll_once()

    assert detected.status == "detected"
    assert runner.calls == [{METRIC: TimeSeries([6.0], [6.0])}]
    assert monitor.state.last_step == 6
    assert no_new.status == "no_new_data"
    assert len(runner.calls) == 1


def test_insufficient_batches_are_retained_then_detected_once_and_cleared(
    tmp_path: Path,
):
    reader = FakeReader(
        "step:1 - timing_s/step:101.0\n",
        "step:1 - timing_s/step:101.0\n",
        "step:1 - timing_s/step:101.0\n"
        "step:2 - timing_s/step:102.0\n",
        "step:1 - timing_s/step:101.0\n"
        "step:2 - timing_s/step:102.0\n"
        "step:3 - timing_s/step:103.0\n",
    )
    runner = FakeRunner(minimum_samples=3)
    monitor = _make_monitor(tmp_path, reader=reader, runner=runner)
    original_standard = copy.deepcopy(monitor.standard)

    first = monitor.poll_once()
    assert first.status == "waiting_for_data"
    assert monitor.state.last_step == 1
    assert monitor.state.inference_buffer[METRIC] == TimeSeries([1.0], [101.0])

    no_new = monitor.poll_once()
    assert no_new.status == "no_new_data"
    assert monitor.state.last_step == 1
    assert monitor.state.inference_buffer[METRIC] == TimeSeries([1.0], [101.0])
    assert runner.calls == []

    second = monitor.poll_once()
    assert second.status == "waiting_for_data"
    assert monitor.state.last_step == 2
    assert monitor.state.inference_buffer[METRIC] == TimeSeries(
        [1.0, 2.0], [101.0, 102.0]
    )

    detected = monitor.poll_once()

    assert detected.status == "detected"
    assert len(runner.calls) == 1
    assert runner.calls[0] == {
        METRIC: TimeSeries([1.0, 2.0, 3.0], [101.0, 102.0, 103.0])
    }
    assert monitor.standard == original_standard
    assert monitor.state.last_step == 3
    assert monitor.state.inference_buffer == {METRIC: TimeSeries()}
    assert monitor.state.buffered_steps == []
    assert monitor.state.round_id == 1

    assert detected.output_path is not None
    saved_text = detected.output_path.read_text(encoding="utf-8")
    saved = json.loads(saved_text)
    metadata = saved["metadata"]
    assert metadata["newStepRange"] == {
        "startStep": 3,
        "endStep": 3,
        "count": 1,
    }
    assert metadata["bufferStepRange"] == {
        "startStep": 1,
        "endStep": 3,
        "count": 3,
    }
    assert metadata["lastStepBefore"] == 2
    assert metadata["lastStepAfter"] == 3
    assert metadata["perMetricPointCounts"][METRIC] == {
        "standard": 3,
        "inference": 3,
    }
    assert saved["debugDetails"] == {
        "metrics": {
            METRIC: {
                "alpha": 0.0123456789,
                "normalModels": [
                    {
                        "rawLowerThreshold": 1.23456789,
                        "rawUpperThreshold": 9.87654321,
                    }
                ],
            }
        }
    }
    assert "private-training-host.internal" not in saved_text
    assert "monitor-user" not in saved_text
    assert "REMOTE_MONITOR_SECRET" not in saved_text


def test_last_step_is_not_committed_until_atomic_save_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    snapshot = "step:1 - timing_s/step:101.0\n"
    reader = FakeReader(snapshot, snapshot)
    runner = FakeRunner(minimum_samples=1)
    monitor = _make_monitor(tmp_path, reader=reader, runner=runner)
    real_save = remote_monitor_module._save_result_atomically
    save_calls = 0

    def flaky_save(*args: Any, **kwargs: Any) -> Path:
        nonlocal save_calls
        save_calls += 1
        if save_calls == 1:
            raise OSError("simulated disk failure")
        return real_save(*args, **kwargs)

    monkeypatch.setattr(remote_monitor_module, "_save_result_atomically", flaky_save)

    failed = monitor.poll_once()

    assert failed.status == "error"
    assert failed.error == {"type": "OSError", "message": "simulated disk failure"}
    assert monitor.state.last_step is None
    assert monitor.state.inference_buffer == {METRIC: TimeSeries()}
    assert monitor.state.buffered_steps == []
    assert monitor.state.round_id == 0

    retried = monitor.poll_once()

    assert retried.status == "detected"
    assert monitor.state.last_step == 1
    assert monitor.state.round_id == 1
    assert len(runner.calls) == 2


def test_ssh_error_does_not_change_existing_state(tmp_path: Path):
    reader = FakeReader(
        "step:1 - timing_s/step:101.0\n",
        RemoteConnectionError("temporary SSH outage"),
    )
    runner = FakeRunner(minimum_samples=2)
    monitor = _make_monitor(tmp_path, reader=reader, runner=runner)
    assert monitor.poll_once().status == "waiting_for_data"
    state_before_error = copy.deepcopy(monitor.state)

    outcome = monitor.poll_once()

    assert outcome.status == "error"
    assert outcome.error == {
        "type": "RemoteConnectionError",
        "message": "temporary SSH outage",
    }
    assert monitor.state == state_before_error
    assert runner.calls == []


def test_unterminated_tail_is_deferred_until_a_later_snapshot(tmp_path: Path):
    reader = FakeReader(
        "step:1 - timing_s/step:101.0",
        "step:1 - timing_s/step:101.0\n",
    )
    runner = FakeRunner(minimum_samples=1)
    monitor = _make_monitor(tmp_path, reader=reader, runner=runner)

    deferred = monitor.poll_once()

    assert deferred.status == "no_new_data"
    assert monitor.state.last_step is None
    assert runner.calls == []

    detected = monitor.poll_once()

    assert detected.status == "detected"
    assert runner.calls == [{METRIC: TimeSeries([1.0], [101.0])}]
    assert monitor.state.last_step == 1


def test_config_uses_environment_variable_name_without_loading_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("RL_MONITOR_PASSWORD", "not-stored-in-config")
    config_path = tmp_path / "remote-monitor.yaml"
    config_path.write_text(
        "standard_log: fixtures/healthy.log\n"
        "remote:\n"
        "  host: trainer.internal\n"
        "  port: 2222\n"
        "  username: observer\n"
        "  log_path: /srv/training/worker.log\n"
        "  key_filename: ssh/id_monitor\n"
        "  known_hosts: ssh/known_hosts\n"
        "  password_env: RL_MONITOR_PASSWORD\n"
        "metrics:\n"
        "  - timing_s/step\n"
        "  - timing_s/step\n"
        "poll_interval_seconds: 15\n"
        "output_dir: results\n",
        encoding="utf-8",
    )

    config = load_remote_monitor_config(config_path)

    assert config.standard_log == (tmp_path / "fixtures/healthy.log").resolve()
    assert config.output_dir == (tmp_path / "results").resolve()
    assert config.metrics == (METRIC,)
    assert config.remote.port == 2222
    assert config.remote.key_filename == (tmp_path / "ssh/id_monitor").resolve()
    assert config.remote.known_hosts == (tmp_path / "ssh/known_hosts").resolve()
    assert config.remote.password_env == "RL_MONITOR_PASSWORD"
    assert "not-stored-in-config" not in repr(config)
    assert "password" not in config.remote.__dataclass_fields__
