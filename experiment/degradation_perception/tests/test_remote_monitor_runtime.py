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
from types import SimpleNamespace
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
REMOTE_PATH = "/srv/private/trainer.log"


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
        self.snapshot_calls = 0
        self.restore_calls = 0

    def minimum_inference_samples(self) -> dict[str, int]:
        return {METRIC: self.minimum_samples}

    def snapshot_runtime_state(self) -> dict[str, int]:
        self.snapshot_calls += 1
        return {'snapshot': self.snapshot_calls}

    def restore_runtime_state(self, snapshot: Any) -> None:
        assert snapshot == {'snapshot': self.snapshot_calls}
        self.restore_calls += 1

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


def _fake_baseline() -> SimpleNamespace:
    return SimpleNamespace(
        baseline_id="monitor-baseline",
        source_type="training_log",
        metrics={METRIC: SimpleNamespace(standard_samples=30)},
    )


def _make_config(tmp_path: Path, *, window_steps: int = 3) -> RemoteMonitorConfig:
    return RemoteMonitorConfig(
        baseline_model=tmp_path / "models" / "baseline_model.json",
        remote=RemoteConnectionConfig(
            host="private-training-host.internal",
            username="monitor-user",
            log_path=REMOTE_PATH,
            password_env="REMOTE_MONITOR_SECRET",
        ),
        metrics=(METRIC,),
        poll_interval_seconds=30,
        inference_window_steps=window_steps,
        state_file=tmp_path / "monitor-output" / "monitor_state.json",
        output_dir=tmp_path / "monitor-output",
        task_id="remote-runtime-test",
    )


def _make_monitor(
    tmp_path: Path,
    *,
    reader: FakeReader,
    runner: FakeRunner,
    window_steps: int = 3,
) -> RemoteMonitor:
    return RemoteMonitor(
        _make_config(tmp_path, window_steps=window_steps),
        reader=reader,
        baseline=_fake_baseline(),
        runner=runner,
        now=lambda: FIXED_NOW,
    )


def test_monitor_loads_baseline_and_runner_once_without_healthy_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = _fake_baseline()
    runner = FakeRunner(minimum_samples=1)
    load_calls: list[tuple[Path, tuple[str, ...]]] = []
    runner_calls: list[tuple[Any, tuple[str, ...], str]] = []

    def fake_load(path: Path, *, expected_metrics: tuple[str, ...]):
        load_calls.append((path, expected_metrics))
        return baseline

    def fake_runner(
        loaded: Any,
        *,
        metrics: tuple[str, ...],
        task_id: str,
    ) -> FakeRunner:
        runner_calls.append((loaded, metrics, task_id))
        return runner

    monkeypatch.setattr(remote_monitor_module, "load_baseline_model", fake_load)
    monkeypatch.setattr(
        remote_monitor_module,
        "BaselineDetectionRunner",
        fake_runner,
    )
    reader = FakeReader(
        "step:1 - timing_s/step:101.0\n",
        "step:1 - timing_s/step:101.0\n",
    )
    config = _make_config(tmp_path)

    monitor = RemoteMonitor(config, reader=reader, now=lambda: FIXED_NOW)
    assert monitor.poll_once().status == "detected"
    assert monitor.poll_once().status == "no_new_data"

    assert load_calls == [(config.baseline_model, (METRIC,))]
    assert runner_calls == [(baseline, (METRIC,), "remote-runtime-test")]
    assert len(runner.calls) == 1
    assert not hasattr(remote_monitor_module, "load_verl_training_log")


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
    assert monitor.state.inference_buffer[METRIC] == TimeSeries([6.0], [6.0])
    assert no_new.status == "no_new_data"
    assert len(runner.calls) == 1


def test_buffer_is_retained_bounded_and_no_new_data_does_not_redetect(
    tmp_path: Path,
):
    reader = FakeReader(
        "step:1 - timing_s/step:101.0\n"
        "step:2 - timing_s/step:102.0\n",
        "step:1 - timing_s/step:101.0\n"
        "step:2 - timing_s/step:102.0\n"
        "step:3 - timing_s/step:103.0\n"
        "step:4 - timing_s/step:104.0\n",
        "step:1 - timing_s/step:101.0\n"
        "step:2 - timing_s/step:102.0\n"
        "step:3 - timing_s/step:103.0\n"
        "step:4 - timing_s/step:104.0\n",
        "step:1 - timing_s/step:101.0\n"
        "step:2 - timing_s/step:102.0\n"
        "step:3 - timing_s/step:103.0\n"
        "step:4 - timing_s/step:104.0\n"
        "step:5 - timing_s/step:105.0\n"
        "step:6 - timing_s/step:106.0\n",
    )
    runner = FakeRunner(minimum_samples=3)
    monitor = _make_monitor(tmp_path, reader=reader, runner=runner)

    waiting = monitor.poll_once()
    assert waiting.status == "waiting_for_data"
    assert monitor.state.buffered_steps == [1, 2]

    first = monitor.poll_once()
    no_new = monitor.poll_once()
    second = monitor.poll_once()

    assert first.status == "detected"
    assert no_new.status == "no_new_data"
    assert second.status == "detected"
    assert runner.calls == [
        {METRIC: TimeSeries([2.0, 3.0, 4.0], [102.0, 103.0, 104.0])},
        {METRIC: TimeSeries([4.0, 5.0, 6.0], [104.0, 105.0, 106.0])},
    ]
    assert monitor.state.last_step == 6
    assert monitor.state.buffered_steps == [4, 5, 6]
    assert monitor.state.inference_buffer[METRIC] == TimeSeries(
        [4.0, 5.0, 6.0],
        [104.0, 105.0, 106.0],
    )
    assert monitor.state.round_id == 2

    assert second.output_path is not None
    saved_text = second.output_path.read_text(encoding="utf-8")
    saved = json.loads(saved_text)
    json.dumps(saved, allow_nan=False)
    metadata = saved["metadata"]
    assert metadata["newStepRange"] == {
        "startStep": 5,
        "endStep": 6,
        "count": 2,
    }
    assert metadata["bufferStepRange"] == {
        "startStep": 4,
        "endStep": 6,
        "count": 3,
    }
    assert metadata["lastStepBefore"] == 4
    assert metadata["lastStepAfter"] == 6
    assert metadata["baselineId"] == "monitor-baseline"
    assert metadata["inferenceWindowSteps"] == 3
    assert metadata["perMetricPointCounts"][METRIC] == {
        "standard": 30,
        "inference": 3,
    }
    assert "private-training-host.internal" not in saved_text
    assert "monitor-user" not in saved_text
    assert "REMOTE_MONITOR_SECRET" not in saved_text


def test_waiting_buffer_and_last_step_are_restored_together_after_restart(
    tmp_path: Path,
):
    first_runner = FakeRunner(minimum_samples=3)
    first = _make_monitor(
        tmp_path,
        reader=FakeReader(
            "step:1 - timing_s/step:101.0\n"
            "step:2 - timing_s/step:102.0\n"
        ),
        runner=first_runner,
    )

    assert first.poll_once().status == "waiting_for_data"
    state_path = first.config.state_file
    on_disk = json.loads(state_path.read_text(encoding="utf-8"))
    assert on_disk["lastStep"] == 2
    assert on_disk["remoteLogPath"] == REMOTE_PATH
    assert on_disk["bufferedSteps"] == [1, 2]
    assert on_disk["inferenceBuffer"][METRIC] == {
        "timestamps": [1.0, 2.0],
        "values": [101.0, 102.0],
    }

    second_runner = FakeRunner(minimum_samples=3)
    restarted = _make_monitor(
        tmp_path,
        reader=FakeReader(
            "step:1 - timing_s/step:101.0\n"
            "step:2 - timing_s/step:102.0\n"
            "step:3 - timing_s/step:103.0\n"
        ),
        runner=second_runner,
    )

    assert restarted.state.last_step == 2
    assert restarted.state.inference_buffer[METRIC] == TimeSeries(
        [1.0, 2.0],
        [101.0, 102.0],
    )
    detected = restarted.poll_once()

    assert detected.status == "detected"
    assert second_runner.calls == [
        {
            METRIC: TimeSeries(
                [1.0, 2.0, 3.0],
                [101.0, 102.0, 103.0],
            )
        }
    ]
    assert restarted.state.last_step == 3
    assert restarted.state.buffered_steps == [1, 2, 3]
    assert restarted.state.round_id == 1


def test_state_save_failure_does_not_commit_waiting_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = FakeRunner(minimum_samples=2)
    monitor = _make_monitor(
        tmp_path,
        reader=FakeReader("step:1 - timing_s/step:101.0\n"),
        runner=runner,
    )

    def fail_state(*args: Any, **kwargs: Any) -> None:
        raise OSError("simulated state disk failure")

    monkeypatch.setattr(
        remote_monitor_module,
        "_save_monitor_state_atomically",
        fail_state,
    )
    outcome = monitor.poll_once()

    assert outcome.status == "error"
    assert outcome.error == {
        "type": "OSError",
        "message": "simulated state disk failure",
    }
    assert monitor.state.last_step is None
    assert monitor.state.inference_buffer == {METRIC: TimeSeries()}
    assert monitor.state.buffered_steps == []
    assert runner.calls == []


def test_result_save_failure_does_not_commit_detected_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = FakeRunner(minimum_samples=1)
    monitor = _make_monitor(
        tmp_path,
        reader=FakeReader("step:1 - timing_s/step:101.0\n"),
        runner=runner,
    )

    def fail_result(*args: Any, **kwargs: Any) -> Path:
        raise OSError("simulated result disk failure")

    monkeypatch.setattr(remote_monitor_module, "_save_result_atomically", fail_result)
    outcome = monitor.poll_once()

    assert outcome.status == "error"
    assert outcome.error == {
        "type": "OSError",
        "message": "simulated result disk failure",
    }
    assert monitor.state.last_step is None
    assert monitor.state.inference_buffer == {METRIC: TimeSeries()}
    assert monitor.state.round_id == 0
    assert not monitor.config.state_file.exists()
    assert len(runner.calls) == 1


def test_state_is_written_after_result_and_failure_keeps_memory_uncommitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = FakeRunner(minimum_samples=1)
    monitor = _make_monitor(
        tmp_path,
        reader=FakeReader("step:1 - timing_s/step:101.0\n"),
        runner=runner,
    )

    def fail_state(*args: Any, **kwargs: Any) -> None:
        assert list(monitor.config.output_dir.glob("*.json"))
        raise OSError("state failed after result")

    monkeypatch.setattr(
        remote_monitor_module,
        "_save_monitor_state_atomically",
        fail_state,
    )
    outcome = monitor.poll_once()

    assert outcome.status == "error"
    assert outcome.error == {
        "type": "OSError",
        "message": "state failed after result",
    }
    assert monitor.state.last_step is None
    assert monitor.state.inference_buffer == {METRIC: TimeSeries()}
    result_files = list(monitor.config.output_dir.glob("*.json"))
    assert [path.name for path in result_files] == [
        "remote-runtime-test_steps_1_1.json"
    ]


def test_ssh_error_does_not_change_existing_persisted_state(tmp_path: Path):
    reader = FakeReader(
        "step:1 - timing_s/step:101.0\n",
        RemoteConnectionError("temporary SSH outage"),
    )
    runner = FakeRunner(minimum_samples=2)
    monitor = _make_monitor(tmp_path, reader=reader, runner=runner)
    assert monitor.poll_once().status == "waiting_for_data"
    state_before_error = copy.deepcopy(monitor.state)
    state_text_before_error = monitor.config.state_file.read_text(encoding="utf-8")

    outcome = monitor.poll_once()

    assert outcome.status == "error"
    assert outcome.error == {
        "type": "RemoteConnectionError",
        "message": "temporary SSH outage",
    }
    assert monitor.state == state_before_error
    assert (
        monitor.config.state_file.read_text(encoding="utf-8")
        == state_text_before_error
    )
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


def test_state_for_another_remote_log_is_rejected(tmp_path: Path):
    config = _make_config(tmp_path)
    config.state_file.parent.mkdir(parents=True)
    config.state_file.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "lastStep": 100,
                "remoteLogPath": "/another/log",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="remoteLogPath"):
        RemoteMonitor(
            config,
            reader=FakeReader(""),
            baseline=_fake_baseline(),
            runner=FakeRunner(minimum_samples=1),
        )


def test_config_uses_baseline_window_and_state_without_loading_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("RL_MONITOR_PASSWORD", "not-stored-in-config")
    config_path = tmp_path / "remote-monitor.yaml"
    policy_path = tmp_path / "algorithm.yaml"
    policy_path.write_text("alpha: 0.01\n", encoding="utf-8")
    config_path.write_text(
        "baseline_model: models/baseline_model.json\n"
        "policy_config: algorithm.yaml\n"
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
        "inference_window_steps: 100\n"
        "state_file: results/monitor_state.json\n"
        "output_dir: results\n",
        encoding="utf-8",
    )

    config = load_remote_monitor_config(config_path)

    assert config.baseline_model == (
        tmp_path / "models/baseline_model.json"
    ).resolve()
    assert config.state_file == (tmp_path / "results/monitor_state.json").resolve()
    assert config.output_dir == (tmp_path / "results").resolve()
    assert config.policy_config == policy_path.resolve()
    assert config.inference_window_steps == 100
    assert config.metrics == (METRIC,)
    assert config.remote.port == 2222
    assert config.remote.key_filename == (tmp_path / "ssh/id_monitor").resolve()
    assert config.remote.known_hosts == (tmp_path / "ssh/known_hosts").resolve()
    assert config.remote.password_env == "RL_MONITOR_PASSWORD"
    assert not hasattr(config, "standard_log")
    assert "not-stored-in-config" not in repr(config)
    assert "password" not in config.remote.__dataclass_fields__


@pytest.mark.parametrize("window", [0, -1, 1.5, True, "not-an-int"])
def test_config_rejects_invalid_inference_window(tmp_path: Path, window: Any):
    config_path = tmp_path / "remote-monitor.yaml"
    config_path.write_text(
        "baseline_model: baseline.json\n"
        "remote:\n"
        "  host: trainer.internal\n"
        "  username: observer\n"
        "  log_path: /srv/training/worker.log\n"
        "metrics:\n"
        "  - timing_s/step\n"
        f"inference_window_steps: {json.dumps(window)}\n"
        "state_file: output/state.json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inference_window_steps"):
        load_remote_monitor_config(config_path)
