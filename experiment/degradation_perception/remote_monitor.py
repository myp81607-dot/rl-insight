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

"""Poll a remote VeRL log over SSH and run the shared degradation detector."""

from __future__ import annotations

import argparse
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

from .detection_runtime import DetectionRunner, serialize_result
from .perception_config import TimeSeries
from .policy import resolve_metric_policy
from .timeseries import preprocess_time_series
from .training_log import (
    load_verl_training_log,
    parse_verl_training_log_with_metadata,
)


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
    standard_log: Path
    remote: RemoteConnectionConfig
    metrics: tuple[str, ...]
    poll_interval_seconds: float
    output_dir: Path
    task_id: str = "degradation-remote-monitor"


@dataclass
class MonitorState:
    """Process-local state committed only at safe poll boundaries."""

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
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive number")
    return number


def _local_path(base: Path, value: Any, name: str) -> Path:
    text = _required_text(value, name)
    path = Path(text).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_remote_monitor_config(path: str | Path) -> RemoteMonitorConfig:
    """Load the documented credential-free YAML shape."""

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
        or any(not isinstance(metric, str) or not metric.strip() for metric in raw_metrics)
    ):
        raise ValueError("metrics must be a non-empty list of metric names")
    metrics = tuple(dict.fromkeys(metric.strip() for metric in raw_metrics))

    try:
        port = int(remote_raw.get("port", 22))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("remote.port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("remote.port must be between 1 and 65535")

    def optional_path(name: str) -> Path | None:
        value = remote_raw.get(name)
        return None if value in (None, "") else _local_path(base, value, f"remote.{name}")

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
            else _required_text(remote_raw.get("password_env"), "remote.password_env")
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
    return RemoteMonitorConfig(
        standard_log=_local_path(base, raw.get("standard_log"), "standard_log"),
        remote=remote,
        metrics=metrics,
        poll_interval_seconds=_positive_float(
            raw.get("poll_interval_seconds", 300),
            "poll_interval_seconds",
        ),
        output_dir=_local_path(
            base,
            raw.get("output_dir", "./monitor_output"),
            "output_dir",
        ),
        task_id=_required_text(
            raw.get("task_id", "degradation-remote-monitor"),
            "task_id",
        ),
    )


def _merge_series(left: TimeSeries, right: TimeSeries) -> TimeSeries:
    return preprocess_time_series(
        [*left.timestamps, *right.timestamps],
        [*left.values, *right.values],
    )


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


def _save_result_atomically(
    result: Mapping[str, Any],
    output_dir: Path,
    task_id: str,
    step_range: Mapping[str, int],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{_safe_task_name(task_id)}_steps_"
        f"{step_range['startStep']}_{step_range['endStep']}.json"
    )
    destination = output_dir / filename
    temporary = output_dir / (filename + ".tmp")
    temporary.write_text(serialize_result(result) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


class RemoteMonitor:
    """Stateful poller with a fixed local standard and transactional batches."""

    def __init__(
        self,
        config: RemoteMonitorConfig,
        *,
        reader: RemoteLogReader | None = None,
        runner: DetectionRunner | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.standard = load_verl_training_log(config.standard_log)
        self.runner = runner or DetectionRunner(
            standard=self.standard,
            metrics=config.metrics,
            task_id=config.task_id,
            source_type="training_log",
        )
        for metric in config.metrics:
            minimum = int(resolve_metric_policy(metric)["minimum_standard_points"])
            if len(self.standard.get(metric, TimeSeries())) < minimum:
                raise ValueError(
                    f"standard log has fewer than {minimum} points for {metric!r}"
                )
        self.reader = reader or ParamikoRemoteLogReader(config.remote)
        self.state = MonitorState(
            inference_buffer={metric: TimeSeries() for metric in config.metrics}
        )
        self._now = now or (lambda: datetime.now(timezone.utc))

    def poll_once(self) -> PollOutcome:
        """Read one snapshot; call the detector only for a complete new batch."""

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

        candidate = {
            metric: _merge_series(
                self.state.inference_buffer.get(metric, TimeSeries()),
                new_series[metric],
            )
            for metric in self.config.metrics
        }
        candidate_steps = sorted(set([*self.state.buffered_steps, *new_steps]))
        new_last_step = max(new_steps)
        requirements = self.runner.minimum_inference_samples()
        enough = all(
            len(candidate.get(metric, TimeSeries())) >= requirements[metric]
            for metric in self.config.metrics
        )
        if not enough:
            self.state.inference_buffer = candidate
            self.state.buffered_steps = candidate_steps
            self.state.last_step = new_last_step
            return PollOutcome(status="waiting_for_data")

        before = self.state.last_step
        new_range = _step_range(new_steps)
        buffer_range = _step_range(candidate_steps)
        assert new_range is not None and buffer_range is not None
        try:
            result = self.runner.run(candidate)
            result["metadata"] = {
                "schemaVersion": 1,
                "mode": "ssh-polling",
                "roundId": self.state.round_id + 1,
                "detectedAt": self._now().astimezone(timezone.utc).isoformat(),
                "sourceType": "training_log",
                "newStepRange": new_range,
                "bufferStepRange": buffer_range,
                "lastStepBefore": before,
                "lastStepAfter": new_last_step,
                "selectedMetrics": list(self.config.metrics),
                "perMetricPointCounts": {
                    metric: {
                        "standard": len(self.standard.get(metric, TimeSeries())),
                        "inference": len(candidate.get(metric, TimeSeries())),
                    }
                    for metric in self.config.metrics
                },
                "pollIntervalSeconds": self.config.poll_interval_seconds,
                "errors": [],
            }
            output = _save_result_atomically(
                result,
                self.config.output_dir,
                self.config.task_id,
                buffer_range,
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            return PollOutcome(
                status="error",
                error={"type": type(exc).__name__, "message": str(exc)},
            )

        self.state.last_step = new_last_step
        self.state.inference_buffer = {
            metric: TimeSeries() for metric in self.config.metrics
        }
        self.state.buffered_steps = []
        self.state.round_id += 1
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
        description="Poll a remote VeRL .log over SSH and detect completed batches.",
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
