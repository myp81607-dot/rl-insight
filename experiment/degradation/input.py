"""Read a selected time range directly from a local Prometheus TSDB."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from .series import Sample, SeriesDataError, SeriesId, TimeSeries

DEFAULT_TSDB_DIR = Path.home() / ".rl-insight" / "data" / "prometheus"
DEFAULT_PROMTOOL_ROOT = Path.home() / ".rl-insight" / "services" / "prometheus"

_SAMPLE_LINE = re.compile(r"^(\{.*\})\s+(\S+)\s+(-?\d+)$")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)=("(?:\\.|[^"\\])*")')


class OfflineInputError(ValueError):
    """The local TSDB cannot provide the requested samples."""


def _promtool(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser()
    system = shutil.which("promtool")
    if system:
        return Path(system)
    installed = sorted(DEFAULT_PROMTOOL_ROOT.rglob("promtool"))
    if installed:
        return installed[-1]
    raise OfflineInputError("promtool was not found")


def _label_set(value: str) -> dict[str, str]:
    return {
        match.group(1): json.loads(match.group(2)) for match in _LABEL.finditer(value)
    }


def _parse_dump(output: str) -> tuple[TimeSeries, ...]:
    merged: dict[SeriesId, dict[float, float]] = defaultdict(dict)
    for line in output.splitlines():
        match = _SAMPLE_LINE.match(line.strip())
        if match is None:
            continue
        try:
            identity = SeriesId.from_label_set(_label_set(match.group(1)))
            value = float(match.group(2))
            timestamp = int(match.group(3)) / 1000.0
        except (SeriesDataError, ValueError, json.JSONDecodeError):
            continue
        if math.isfinite(value):
            merged[identity][timestamp] = value
    return tuple(
        TimeSeries(
            identity=identity,
            samples=tuple(
                Sample(timestamp, value) for timestamp, value in sorted(samples.items())
            ),
        )
        for identity, samples in sorted(merged.items())
        if samples
    )


def load_time_series(
    data_dir: Path,
    *,
    start_time: float,
    end_time: float,
    metric_names: frozenset[str],
    promtool_path: Path | None = None,
) -> tuple[TimeSeries, ...]:
    """Dump configured scalar series from the RL-Insight Prometheus TSDB."""

    selector = (
        '{__name__=~"^(' + "|".join(sorted(map(re.escape, metric_names))) + ')$"}'
    )
    command = [
        str(_promtool(promtool_path)),
        "tsdb",
        "dump",
        f"--min-time={int(start_time * 1000)}",
        f"--max-time={int(end_time * 1000)}",
        f"--match={selector}",
        str(data_dir.expanduser()),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise OfflineInputError(completed.stderr.strip() or "promtool tsdb dump failed")
    series = _parse_dump(completed.stdout)
    if not series:
        raise OfflineInputError(
            "the selected time range contains no configured metrics"
        )
    return series


__all__ = ["DEFAULT_TSDB_DIR", "OfflineInputError", "load_time_series"]
