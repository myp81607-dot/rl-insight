"""Small data model shared by offline input and the analysis algorithm."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class SeriesDataError(ValueError):
    """Offline series data is malformed."""


@dataclass(frozen=True, order=True)
class SeriesId:
    """A Prometheus series identified by metric name and labels."""

    name: str
    labels: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_label_set(cls, value: Mapping[str, Any]) -> SeriesId:
        if not isinstance(value, Mapping):
            raise SeriesDataError("series labels must be an object")
        name = value.get("__name__")
        if not isinstance(name, str) or not name:
            raise SeriesDataError("series is missing __name__")
        labels = []
        for key, label_value in value.items():
            if not isinstance(key, str) or not isinstance(label_value, str):
                raise SeriesDataError("series labels must be strings")
            if key != "__name__":
                labels.append((key, label_value))
        return cls(name=name, labels=tuple(sorted(labels)))

    def to_label_set(self) -> dict[str, str]:
        return {"__name__": self.name, **dict(self.labels)}


@dataclass(frozen=True)
class Sample:
    timestamp: float
    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp) or not math.isfinite(self.value):
            raise SeriesDataError("sample timestamp and value must be finite")


@dataclass(frozen=True)
class TimeSeries:
    identity: SeriesId
    samples: tuple[Sample, ...]


__all__ = ["Sample", "SeriesDataError", "SeriesId", "TimeSeries"]
