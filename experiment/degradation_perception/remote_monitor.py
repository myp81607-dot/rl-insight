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

"""Poll a remote VeRL log and detect with one persisted KDE baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml

from .algorithm_policy import load_algorithm_policy_config
from .baseline_model import (
    BaselineDetectionRunner,
    load_baseline_model,
)
from .detection_runtime import serialize_result
from .perception_config import TimeSeries
from .timeseries import preprocess_time_series
from .training_log import parse_verl_training_log_with_metadata


STATE_SCHEMA_VERSION = 1


class RemoteMonitorError(RuntimeError):
    """Base class for actionable remote-monitor errors."""


class RemoteDependencyError(RemoteMonitorError):
    """Raised when the optional SSH dependency is unavailable."""


class RemoteConnectionError(RemoteMonitorError):
    """Raised when the remote snapshot cannot be read safely."""


@dataclass(frozen=True)
class RemoteConnectionConfig:
    host: str
    username: str
    log_path: str
    port: int = 22
    key_filename: Path | None = None
    known_hosts: Path | None = None
    password_env: str | None = None
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 30.0


@dataclass(frozen=True)
class RemoteMonitorConfig:
    baseline_model: Path
    remote: RemoteConnectionConfig
    metrics: tuple[str, ...]
    poll_interval_seconds: float
    inference_window_steps: int
    state_file: Path
    output_dir: Path
    task_id: str = "degradation-remote-monitor"
    policy_config: Path | None = None


@dataclass
class MonitorState:
    """Durable state committed only after all work in a poll succeeds."""

    last_step: int | None = None
    inference_buffer: dict[str, TimeSeries] = field(default_factory=dict)
    buffered_steps: list[int] = field(default_factory=list)
    round_id: int = 0


@dataclass(frozen=True)
class PollOutcome:
    status: str
    result: dict[str, Any] | None = None
    output_path: Path | None = None
    error: dict[str, str] | None = None


class RemoteLogReader(Protocol):
    def read_snapshot(self) -> str:
        """Return a read-only UTF-8 snapshot of the remote log."""


class ModelDetectionRunner(Protocol):
    def minimum_inference_samples(self) -> dict[str, int]:
        """Return the loaded model's per-metric inference minimum."""

    def snapshot_runtime_state(self) -> Any: ...

    def restore_runtime_state(self, snapshot: Any) -> None: ...

    def run(self, inference: Mapping[str, TimeSeries]) -> dict[str, Any]:
        """Detect one rolling inference window without fitting KDE."""


class ParamikoRemoteLogReader:
    """Read a remote file through Paramiko SFTP without modifying it."""

    def __init__(self, config: RemoteConnectionConfig) -> None:
        self.config = config

    def read_snapshot(self) -> str:
        try:
            import paramiko
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise RemoteDependencyError(
                "Paramiko is required for SSH polling; install "
                "experiment/degradation_perception/requirements-remote.txt"
            ) from exc

        client: Any = None
        sftp: Any = None
        remote_file: Any = None
        try:
            client = paramiko.SSHClient()
            if self.config.known_hosts is not None:
                client.load_host_keys(str(self.config.known_hosts))
            else:
                client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
            connect_kwargs: dict[str, Any] = {
                "hostname": self.config.host,
                "port": self.config.port,
                "username": self.config.username,
                "timeout": self.config.connect_timeout_seconds,
                "banner_timeout": self.config.connect_timeout_seconds,
                "auth_timeout": self.config.connect_timeout_seconds,
                "look_for_keys": True,
                "allow_agent": True,
            }
            if self.config.key_filename is not None:
                connect_kwargs["key_filename"] = str(self.config.key_filename)
            if self.config.password_env is not None:
                password = os.environ.get(self.config.password_env)
                if password is None:
                    raise RemoteConnectionError(
                        "configured SSH password environment variable is unset"
                    )
                connect_kwargs["password"] = password
            client.connect(**connect_kwargs)
            sftp = client.open_sftp()
            remote_file = sftp.open(self.config.log_path, "rb")
            if hasattr(remote_file, "settimeout"):
                remote_file.settimeout(self.config.read_timeout_seconds)
            raw = remote_file.read()
            if isinstance(raw, str):
                return raw
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="replace")
            raise RemoteConnectionError("remote log reader returned non-text data")
        except KeyboardInterrupt:
            raise
        except RemoteMonitorError:
            raise
        except Exception as exc:
            raise RemoteConnectionError(
                f"SSH log read failed ({type(exc).__name__}): {exc}"
            ) from exc
        finally:
            for resource in (remote_file, sftp, client):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception:
                        pass


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive number")
    return number


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
        original = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not math.isfinite(original) or original != number or number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _local_path(base: Path, value: Any, name: str) -> Path:
    text = _required_text(value, name)
    path = Path(text).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_remote_monitor_config(path: str | Path) -> RemoteMonitorConfig:
    """Load the credential-free model-backed monitor YAML."""

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"could not load monitor config: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("monitor config root must be an object")
    base = config_path.resolve().parent
    remote_raw = raw.get("remote")
    if not isinstance(remote_raw, Mapping):
        raise ValueError("remote must be an object")
    raw_metrics = raw.get("metrics")
    if (
        not isinstance(raw_metrics, list)
        or not raw_metrics
        or any(
            not isinstance(metric, str) or not metric.strip()
            for metric in raw_metrics
        )
    ):
        raise ValueError("metrics must be a non-empty list of metric names")
    metrics = tuple(dict.fromkeys(metric.strip() for metric in raw_metrics))

    try:
        port = int(remote_raw.get("port", 22))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("remote.port must be an integer") from exc
    if isinstance(remote_raw.get("port", 22), bool) or not 1 <= port <= 65535:
        raise ValueError("remote.port must be between 1 and 65535")

    def optional_path(name: str) -> Path | None:
        value = remote_raw.get(name)
        return None if value in (None, "") else _local_path(
            base,
            value,
            f"remote.{name}",
        )

    remote = RemoteConnectionConfig(
        host=_required_text(remote_raw.get("host"), "remote.host"),
        port=port,
        username=_required_text(remote_raw.get("username"), "remote.username"),
        log_path=_required_text(remote_raw.get("log_path"), "remote.log_path"),
        key_filename=optional_path("key_filename"),
        known_hosts=optional_path("known_hosts"),
        password_env=(
            None
            if remote_raw.get("password_env") in (None, "")
            else _required_text(
                remote_raw.get("password_env"),
                "remote.password_env",
            )
        ),
        connect_timeout_seconds=_positive_float(
            remote_raw.get("connect_timeout_seconds", 10),
            "remote.connect_timeout_seconds",
        ),
        read_timeout_seconds=_positive_float(
            remote_raw.get("read_timeout_seconds", 30),
            "remote.read_timeout_seconds",
        ),
    )
    output_dir = _local_path(
        base,
        raw.get("output_dir", "./monitor_output"),
        "output_dir",
    )
    policy_config = (
        None
        if raw.get("policy_config") in (None, "")
        else _local_path(base, raw.get("policy_config"), "policy_config")
    )
    if policy_config is not None and not policy_config.is_file():
        raise ValueError(
            f"policy_config is not an existing file: {policy_config}"
        )
    return RemoteMonitorConfig(
        baseline_model=_local_path(
            base,
            raw.get("baseline_model"),
            "baseline_model",
        ),
        remote=remote,
        metrics=metrics,
        poll_interval_seconds=_positive_float(
            raw.get("poll_interval_seconds", 300),
            "poll_interval_seconds",
        ),
        inference_window_steps=_positive_int(
            raw.get("inference_window_steps"),
            "inference_window_steps",
        ),
        state_file=_local_path(base, raw.get("state_file"), "state_file"),
        output_dir=output_dir,
        task_id=_required_text(
            raw.get("task_id", "degradation-remote-monitor"),
            "task_id",
        ),
        policy_config=policy_config,
    )


def _merge_series(left: TimeSeries, right: TimeSeries) -> TimeSeries:
    return preprocess_time_series(
        [*left.timestamps, *right.timestamps],
        [*left.values, *right.values],
    )


def _copy_series(series: TimeSeries) -> TimeSeries:
    return TimeSeries(list(series.timestamps), list(series.values))


def _copy_buffer(buffer: Mapping[str, TimeSeries]) -> dict[str, TimeSeries]:
    return {metric: _copy_series(series) for metric, series in buffer.items()}


def _complete_lines(snapshot: str) -> str:
    """Ignore an unterminated active-log tail until the next poll."""

    if not snapshot or snapshot.endswith(("\n", "\r")):
        return snapshot
    last_break = max(snapshot.rfind("\n"), snapshot.rfind("\r"))
    return "" if last_break < 0 else snapshot[: last_break + 1]


def _step_range(steps: Sequence[int]) -> dict[str, int] | None:
    unique = sorted(set(int(step) for step in steps))
    if not unique:
        return None
    return {
        "startStep": unique[0],
        "endStep": unique[-1],
        "count": len(unique),
    }


def _safe_task_name(task_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("._")
    return value or "degradation"


def _write_text_atomically(destination: Path, text: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _save_result_atomically(
    result: Mapping[str, Any],
    output_dir: Path,
    task_id: str,
    step_range: Mapping[str, int],
) -> Path:
    filename = (
        f"{_safe_task_name(task_id)}_steps_"
        f"{step_range['startStep']}_{step_range['endStep']}.json"
    )
    destination = output_dir / filename
    _write_text_atomically(destination, serialize_result(result) + "\n")
    return destination


def _state_payload(
    state: MonitorState,
    *,
    remote_log_path: str,
    metrics: Sequence[str],
) -> dict[str, Any]:
    return {
        "schemaVersion": STATE_SCHEMA_VERSION,
        "lastStep": state.last_step,
        "remoteLogPath": remote_log_path,
        "roundId": state.round_id,
        "metrics": list(metrics),
        "bufferedSteps": list(state.buffered_steps),
        "inferenceBuffer": {
            metric: {
                "timestamps": list(
                    state.inference_buffer.get(metric, TimeSeries()).timestamps
                ),
                "values": list(
                    state.inference_buffer.get(metric, TimeSeries()).values
                ),
            }
            for metric in metrics
        },
    }


def _save_monitor_state_atomically(
    state: MonitorState,
    path: Path,
    *,
    remote_log_path: str,
    metrics: Sequence[str],
) -> None:
    encoded = json.dumps(
        _state_payload(
            state,
            remote_log_path=remote_log_path,
            metrics=metrics,
        ),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    )
    _write_text_atomically(path, encoded + "\n")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"state file contains non-finite JSON number: {value}")


def _state_series(value: Any, *, metric: str) -> TimeSeries:
    if not isinstance(value, Mapping):
        raise ValueError(f"state inferenceBuffer[{metric!r}] must be an object")
    timestamps = value.get("timestamps")
    values = value.get("values")
    if not isinstance(timestamps, list) or not isinstance(values, list):
        raise ValueError(
            f"state inferenceBuffer[{metric!r}] must contain array fields"
        )
    if len(timestamps) != len(values):
        raise ValueError(
            f"state inferenceBuffer[{metric!r}] timestamps/values length mismatch"
        )
    for timestamp, item in zip(timestamps, values):
        if isinstance(timestamp, bool) or isinstance(item, bool):
            raise ValueError(
                f"state inferenceBuffer[{metric!r}] contains boolean data"
            )
        try:
            numeric_timestamp = float(timestamp)
            numeric_value = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"state inferenceBuffer[{metric!r}] contains non-numeric data"
            ) from exc
        if (
            not math.isfinite(numeric_timestamp)
            or numeric_timestamp != int(numeric_timestamp)
            or not math.isfinite(numeric_value)
        ):
            raise ValueError(
                f"state inferenceBuffer[{metric!r}] contains invalid step data"
            )
    series = preprocess_time_series(timestamps, values)
    if len(series) != len(timestamps):
        raise ValueError(
            f"state inferenceBuffer[{metric!r}] contains duplicate steps"
        )
    return series


def _load_monitor_state(
    path: Path,
    *,
    remote_log_path: str,
    metrics: Sequence[str],
    window_steps: int,
) -> MonitorState:
    empty = {metric: TimeSeries() for metric in metrics}
    if not path.exists():
        return MonitorState(inference_buffer=empty)
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"could not load monitor state: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("monitor state root must be an object")
    if raw.get("schemaVersion", STATE_SCHEMA_VERSION) != STATE_SCHEMA_VERSION:
        raise ValueError("monitor state schemaVersion must be 1")
    if raw.get("remoteLogPath") != remote_log_path:
        raise ValueError(
            "monitor state remoteLogPath does not match configured remote log"
        )
    state_metrics = raw.get("metrics")
    if state_metrics is not None and state_metrics != list(metrics):
        raise ValueError("monitor state metrics do not match configured metrics")

    last_step = raw.get("lastStep")
    if last_step is not None and (
        isinstance(last_step, bool) or not isinstance(last_step, int)
    ):
        raise ValueError("monitor state lastStep must be an integer or null")
    round_id = raw.get("roundId", 0)
    if (
        isinstance(round_id, bool)
        or not isinstance(round_id, int)
        or round_id < 0
    ):
        raise ValueError("monitor state roundId must be a non-negative integer")
    raw_buffer = raw.get("inferenceBuffer", {})
    if not isinstance(raw_buffer, Mapping):
        raise ValueError("monitor state inferenceBuffer must be an object")
    buffer = {
        metric: _state_series(raw_buffer[metric], metric=metric)
        if metric in raw_buffer
        else TimeSeries()
        for metric in metrics
    }
    buffer, buffered_steps = _trim_to_window(buffer, window_steps)
    if last_step is not None and any(step > last_step for step in buffered_steps):
        raise ValueError("monitor state buffer contains a step above lastStep")
    return MonitorState(
        last_step=last_step,
        inference_buffer=buffer,
        buffered_steps=buffered_steps,
        round_id=round_id,
    )


def _trim_to_window(
    buffer: Mapping[str, TimeSeries],
    window_steps: int,
) -> tuple[dict[str, TimeSeries], list[int]]:
    all_steps = sorted(
        {
            int(timestamp)
            for series in buffer.values()
            for timestamp in series.timestamps
        }
    )
    retained_steps = all_steps[-window_steps:]
    retained = set(retained_steps)
    trimmed: dict[str, TimeSeries] = {}
    for metric, series in buffer.items():
        pairs = [
            (timestamp, value)
            for timestamp, value in zip(series.timestamps, series.values)
            if int(timestamp) in retained
        ]
        trimmed[metric] = TimeSeries(
            timestamps=[timestamp for timestamp, _ in pairs],
            values=[value for _, value in pairs],
        )
    return trimmed, retained_steps


def _baseline_identity(baseline: Any, name: str, fallback: str) -> str:
    value = getattr(baseline, name, None)
    if value is None and isinstance(baseline, Mapping):
        camel_name = {
            "baseline_id": "baselineId",
            "source_type": "sourceType",
        }.get(name, name)
        value = baseline.get(camel_name)
    return value if isinstance(value, str) and value else fallback


def _standard_sample_count(baseline: Any, metric: str) -> int | None:
    metrics = getattr(baseline, "metrics", None)
    if metrics is None and isinstance(baseline, Mapping):
        metrics = baseline.get("metrics")
    if not isinstance(metrics, Mapping) or metric not in metrics:
        return None
    entry = metrics[metric]
    value = getattr(entry, "standard_samples", None)
    if value is None and isinstance(entry, Mapping):
        value = entry.get("standardSamples")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return None


class RemoteMonitor:
    """Stateful poller backed by one immutable, pre-fitted KDE model."""

    def __init__(
        self,
        config: RemoteMonitorConfig,
        *,
        reader: RemoteLogReader | None = None,
        baseline: Any | None = None,
        runner: ModelDetectionRunner | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.baseline = (
            load_baseline_model(
                config.baseline_model,
                expected_metrics=config.metrics,
            )
            if baseline is None
            else baseline
        )
        if runner is None:
            if config.policy_config is None:
                self.runner = BaselineDetectionRunner(
                    self.baseline,
                    metrics=config.metrics,
                    task_id=config.task_id,
                )
            else:
                policy = load_algorithm_policy_config(config.policy_config)
                self.runner = BaselineDetectionRunner(
                    self.baseline,
                    metrics=config.metrics,
                    task_id=config.task_id,
                    metric_policies=policy.for_metrics(config.metrics),
                    history_policy=policy.history_policy,
                )
        else:
            self.runner = runner
        requirements = self.runner.minimum_inference_samples()
        for metric in config.metrics:
            minimum = requirements.get(metric)
            if (
                isinstance(minimum, bool)
                or not isinstance(minimum, int)
                or minimum <= 0
            ):
                raise ValueError(
                    f"invalid minimum inference sample count for {metric!r}"
                )
            if minimum > config.inference_window_steps:
                raise ValueError(
                    "inference_window_steps is smaller than the required "
                    f"{minimum} samples for {metric!r}"
                )
        self._minimum_samples = requirements
        self._standard_samples = {
            metric: _standard_sample_count(self.baseline, metric)
            for metric in config.metrics
        }
        self.reader = reader or ParamikoRemoteLogReader(config.remote)
        self.state = _load_monitor_state(
            config.state_file,
            remote_log_path=config.remote.log_path,
            metrics=config.metrics,
            window_steps=config.inference_window_steps,
        )
        self._now = now or (lambda: datetime.now(timezone.utc))

    def _commit_state(self, state: MonitorState) -> None:
        _save_monitor_state_atomically(
            state,
            self.config.state_file,
            remote_log_path=self.config.remote.log_path,
            metrics=self.config.metrics,
        )

    def poll_once(self) -> PollOutcome:
        """Read one snapshot and transactionally process only new steps."""

        try:
            snapshot = self.reader.read_snapshot()
            parsed = parse_verl_training_log_with_metadata(
                _complete_lines(snapshot),
                min_step_exclusive=self.state.last_step,
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            return PollOutcome(
                status="error",
                error={"type": type(exc).__name__, "message": str(exc)},
            )

        new_series = {
            metric: parsed.series.get(metric, TimeSeries())
            for metric in self.config.metrics
        }
        new_steps = sorted(
            {
                int(timestamp)
                for series in new_series.values()
                for timestamp in series.timestamps
            }
        )
        if not new_steps:
            return PollOutcome(status="no_new_data")

        merged = {
            metric: _merge_series(
                self.state.inference_buffer.get(metric, TimeSeries()),
                new_series[metric],
            )
            for metric in self.config.metrics
        }
        candidate_buffer, candidate_steps = _trim_to_window(
            merged,
            self.config.inference_window_steps,
        )
        candidate = MonitorState(
            last_step=max(new_steps),
            inference_buffer=candidate_buffer,
            buffered_steps=candidate_steps,
            round_id=self.state.round_id,
        )
        enough = all(
            len(candidate_buffer.get(metric, TimeSeries()))
            >= self._minimum_samples[metric]
            for metric in self.config.metrics
        )
        if not enough:
            try:
                self._commit_state(candidate)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                return PollOutcome(
                    status="error",
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
            self.state = candidate
            return PollOutcome(status="waiting_for_data")

        before = self.state.last_step
        new_range = _step_range(new_steps)
        buffer_range = _step_range(candidate_steps)
        assert new_range is not None and buffer_range is not None
        runtime_snapshot: Any = None
        snapshot_taken = False
        try:
            runtime_snapshot = self.runner.snapshot_runtime_state()
            snapshot_taken = True
            result = self.runner.run(_copy_buffer(candidate_buffer))
            next_round = self.state.round_id + 1
            candidate.round_id = next_round
            result["metadata"] = {
                "schemaVersion": 1,
                "mode": "ssh-polling",
                "roundId": next_round,
                "detectedAt": self._now().astimezone(timezone.utc).isoformat(),
                "sourceType": _baseline_identity(
                    self.baseline,
                    "source_type",
                    "training_log",
                ),
                "baselineId": _baseline_identity(
                    self.baseline,
                    "baseline_id",
                    "default",
                ),
                "newStepRange": new_range,
                "bufferStepRange": buffer_range,
                "lastStepBefore": before,
                "lastStepAfter": candidate.last_step,
                "selectedMetrics": list(self.config.metrics),
                "perMetricPointCounts": {
                    metric: {
                        "standard": self._standard_samples[metric],
                        "inference": len(
                            candidate_buffer.get(metric, TimeSeries())
                        ),
                    }
                    for metric in self.config.metrics
                },
                "pollIntervalSeconds": self.config.poll_interval_seconds,
                "inferenceWindowSteps": self.config.inference_window_steps,
                "errors": [],
            }
            output = _save_result_atomically(
                result,
                self.config.output_dir,
                self.config.task_id,
                buffer_range,
            )
            self._commit_state(candidate)
        except KeyboardInterrupt:
            if snapshot_taken:
                self.runner.restore_runtime_state(runtime_snapshot)
            raise
        except Exception as exc:
            if snapshot_taken:
                try:
                    self.runner.restore_runtime_state(runtime_snapshot)
                except Exception as rollback_exc:
                    return PollOutcome(
                        status='error',
                        error={
                            'type': type(rollback_exc).__name__,
                            'message': (
                                f'{exc}; detector rollback failed: '
                                f'{rollback_exc}'
                            ),
                        },
                    )
            return PollOutcome(
                status="error",
                error={"type": type(exc).__name__, "message": str(exc)},
            )

        self.state = candidate
        return PollOutcome(status="detected", result=result, output_path=output)

    def run_forever(
        self,
        *,
        max_polls: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        polls = 0
        while max_polls is None or polls < max_polls:
            outcome = self.poll_once()
            polls += 1
            if outcome.status == "error":
                assert outcome.error is not None
                sys.stderr.write(
                    f"poll error: {outcome.error['type']}: "
                    f"{outcome.error['message']}\n"
                )
            elif outcome.output_path is not None:
                sys.stdout.write(f"saved: {outcome.output_path}\n")
            if max_polls is None or polls < max_polls:
                sleep(self.config.poll_interval_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiment.degradation_perception.remote_monitor",
        description=(
            "Poll a remote VeRL .log over SSH and detect with a saved baseline."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Execute one poll for connectivity and configuration checks.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        config = load_remote_monitor_config(args.config)
        monitor = RemoteMonitor(config)
        monitor.run_forever(max_polls=1 if args.once else None)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MonitorState",
    "ParamikoRemoteLogReader",
    "PollOutcome",
    "RemoteConnectionConfig",
    "RemoteConnectionError",
    "RemoteDependencyError",
    "RemoteMonitor",
    "RemoteMonitorConfig",
    "load_remote_monitor_config",
    "main",
]
