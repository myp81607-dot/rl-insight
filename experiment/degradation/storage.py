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

"""Persist frozen baselines and degradation events as JSON."""

from __future__ import annotations

import json
import shutil
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .state_schema import (
    SCHEMA_VERSION,
    StateInspection,
    StateStatus,
    StateValidationError,
    classify_state,
    empty_events_payload,
    validate_baseline_payload,
    validate_events_payload,
)

if TYPE_CHECKING:
    from .association import AssociationItem, AssociationResult
    from .baseline import BaselineParameters, SeriesBaseline
    from .detector import DegradationEvent
    from .prometheus import SeriesId

STANDARD_DATA_NAME = "standard_data.json"
ABNORMAL_DATA_NAME = "abnormal_data.json"


class StorageError(RuntimeError):
    """A degradation state file cannot be read or updated."""


@dataclass(frozen=True)
class BaselineSnapshot:
    """The complete frozen baseline state required after a restart."""

    start_step: int
    end_step: int
    global_step_identity: SeriesId
    parameters: BaselineParameters
    baselines: dict[SeriesId, SeriesBaseline]


def _identity_to_json(identity: SeriesId) -> dict[str, Any]:
    return {"metric": identity.name, "labels": dict(identity.labels)}


def _identity_from_json(value: object) -> SeriesId:
    from .prometheus import SeriesId

    if not isinstance(value, Mapping):
        raise StorageError("series identity must be a JSON object")
    metric = value.get("metric")
    labels = value.get("labels", {})
    if not isinstance(metric, str) or not metric or not isinstance(labels, Mapping):
        raise StorageError("series identity has invalid metric or labels")
    label_set = {"__name__": metric}
    for name, label_value in labels.items():
        if not isinstance(name, str) or not isinstance(label_value, str):
            raise StorageError("series labels must be strings")
        label_set[name] = label_value
    return SeriesId.from_label_set(label_set)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise StorageError(f"cannot write {path}: {exc}") from exc


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise StorageError(f"{path} must contain a JSON object")
    return payload


def _path_present(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise StorageError(f"cannot inspect {path}: {exc}") from exc
    return True


def _validate_parent_directory(path: Path) -> None:
    ancestor = path
    while True:
        try:
            ancestor.lstat()
        except FileNotFoundError:
            parent = ancestor.parent
            if parent == ancestor:
                return
            ancestor = parent
            continue
        except NotADirectoryError:
            ancestor = ancestor.parent
            continue
        except OSError as exc:
            raise StorageError(
                f"cannot inspect state directory {ancestor}: {exc}"
            ) from exc
        try:
            is_directory = ancestor.is_dir()
        except OSError as exc:
            raise StorageError(
                f"cannot inspect state directory {ancestor}: {exc}"
            ) from exc
        if not is_directory:
            raise StorageError(
                f"state directory has a non-directory ancestor: {ancestor}"
            )
        return


def _read_json_for_inspection(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StorageError(f"cannot read {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"cannot decode {path}: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path} must contain a JSON object"
    return payload, None


def _baseline_to_json(snapshot: BaselineSnapshot) -> dict[str, Any]:
    series = []
    for identity, baseline in sorted(snapshot.baselines.items()):
        series.append(
            {
                **_identity_to_json(identity),
                "sample_count": baseline.sample_count,
                "ranges": [asdict(normal_range) for normal_range in baseline.ranges],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline": {
            "start_step": snapshot.start_step,
            "end_step": snapshot.end_step,
            "global_step": _identity_to_json(snapshot.global_step_identity),
            "parameters": asdict(snapshot.parameters),
            "series": series,
        },
    }


def _baseline_from_json(payload: Mapping[str, Any]) -> BaselineSnapshot:
    from .baseline import BaselineParameters, NormalRange, SeriesBaseline

    try:
        validate_baseline_payload(payload)
    except StateValidationError as exc:
        raise StorageError(str(exc)) from exc
    raw = payload.get("baseline")
    assert isinstance(raw, Mapping)
    try:
        start_step = int(raw["start_step"])
        end_step = int(raw["end_step"])
        parameters = BaselineParameters(**dict(raw["parameters"]))
        global_step_identity = _identity_from_json(raw["global_step"])
        raw_series = raw["series"]
    except (KeyError, TypeError, ValueError) as exc:
        raise StorageError("standard_data.json is invalid") from exc
    if not isinstance(raw_series, list) or start_step > end_step:
        raise StorageError("standard_data.json baseline is invalid")

    baselines: dict[SeriesId, SeriesBaseline] = {}
    try:
        for item in raw_series:
            identity = _identity_from_json(item)
            if not isinstance(item, Mapping) or not isinstance(
                item.get("ranges"), list
            ):
                raise TypeError
            ranges = tuple(NormalRange(**dict(value)) for value in item["ranges"])
            baselines[identity] = SeriesBaseline(
                identity=identity,
                sample_count=int(item["sample_count"]),
                ranges=ranges,
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise StorageError("standard_data.json series are invalid") from exc
    if not baselines:
        raise StorageError("standard_data.json contains no baselines")
    return BaselineSnapshot(
        start_step=start_step,
        end_step=end_step,
        global_step_identity=global_step_identity,
        parameters=parameters,
        baselines=baselines,
    )


def _empty_abnormal_data() -> dict[str, Any]:
    return empty_events_payload()


def _load_abnormal_data(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_abnormal_data()
    payload = dict(_read_json(path))
    try:
        validate_events_payload(payload)
    except StateValidationError as exc:
        raise StorageError(str(exc)) from exc
    return payload


def _association_to_json(
    result: AssociationResult,
    top_associations: Sequence[AssociationItem],
    directions: Mapping[SeriesId, str],
) -> dict[str, Any]:
    missing = [
        item.identity for item in top_associations if item.identity not in directions
    ]
    if missing:
        raise StorageError("candidate directions are missing")
    return {
        "status": result.status,
        "top_k": [
            {
                **_identity_to_json(item.identity),
                "direction": directions[item.identity],
                "association_percent": item.association_score * 100.0,
                "pearson": item.pearson,
                "spearman": item.spearman,
                "selected_correlation": item.selected_correlation,
                "selected_correlation_method": item.selected_correlation_method,
                "correlation_share": item.correlation_share,
                "random_forest_eligible": item.random_forest_eligible,
                "random_forest_importance": item.random_forest_importance,
                "random_forest_share": item.random_forest_share,
                "aligned_sample_count": item.aligned_sample_count,
                "coverage_ratio": item.coverage_ratio,
            }
            for item in top_associations
        ],
        "effective_correlation_weight": result.effective_correlation_weight,
        "effective_random_forest_weight": result.effective_random_forest_weight,
        "random_forest_status": result.random_forest_status,
        "random_forest_reason": result.random_forest_reason,
        "random_forest_sample_count": result.random_forest_sample_count,
        "random_forest_validation_score": result.random_forest_validation_score,
        "skipped_candidates": [
            {**_identity_to_json(item.identity), "reason": item.reason}
            for item in result.skipped_candidates
        ],
    }


class JsonStorage:
    """Own the two JSON files used by the v0.1 runtime."""

    def __init__(self, standard_data_path: Path, abnormal_data_path: Path) -> None:
        self.standard_data_path = Path(standard_data_path)
        self.abnormal_data_path = Path(abnormal_data_path)
        if self.standard_data_path == self.abnormal_data_path:
            raise ValueError("standard and abnormal data paths must be different")

    @property
    def state_paths(self) -> tuple[Path, Path]:
        """Return the two configured state paths in stable order."""

        return self.standard_data_path, self.abnormal_data_path

    def inspect_state(self) -> StateInspection:
        """Read and classify existing state without creating or changing files."""

        for path in self.state_paths:
            _validate_parent_directory(path.parent)
        standard_present = _path_present(self.standard_data_path)
        abnormal_present = _path_present(self.abnormal_data_path)
        if not standard_present and not abnormal_present:
            return StateInspection(StateStatus.UNINITIALIZED)
        if not standard_present:
            return StateInspection(
                StateStatus.INVALID_STATE,
                error=(
                    f"{self.abnormal_data_path.name} exists without "
                    f"{self.standard_data_path.name}"
                ),
            )

        payloads: list[dict[str, Any] | None] = []
        for path, present in (
            (self.standard_data_path, standard_present),
            (self.abnormal_data_path, abnormal_present),
        ):
            if not present:
                payloads.append(None)
                continue
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                raise StorageError(f"cannot inspect {path}: {exc}") from exc
            if not stat.S_ISREG(mode):
                return StateInspection(
                    StateStatus.INVALID_STATE,
                    error=f"state path is not a regular file: {path}",
                )
            payload, error = _read_json_for_inspection(path)
            if error is not None:
                return StateInspection(StateStatus.INVALID_STATE, error=error)
            payloads.append(payload)

        baseline_payload, events_payload = payloads
        assert baseline_payload is not None
        return classify_state(
            baseline_present=True,
            baseline_payload=baseline_payload,
            events_present=abnormal_present,
            events_payload=events_payload,
            baseline_source=str(self.standard_data_path),
            events_source=str(self.abnormal_data_path),
        )

    def inspect_state_files(self) -> list[dict[str, Any]]:
        """Return per-file metadata for the readiness check without writes."""

        validators = (validate_baseline_payload, validate_events_payload)
        summaries = []
        for path, validator in zip(self.state_paths, validators):
            _validate_parent_directory(path.parent)
            present = _path_present(path)
            result: dict[str, Any] = {"path": str(path), "exists": present}
            if not present:
                summaries.append(result)
                continue
            try:
                stat_result = path.lstat()
            except OSError as exc:
                raise StorageError(f"cannot inspect {path}: {exc}") from exc
            if not stat.S_ISREG(stat_result.st_mode):
                result.update(
                    valid_json=False, error="state path is not a regular file"
                )
                summaries.append(result)
                continue
            result["size_bytes"] = stat_result.st_size
            payload, error = _read_json_for_inspection(path)
            if error is not None:
                result.update(valid_json=False, error=error)
                summaries.append(result)
                continue
            assert payload is not None
            try:
                validator(payload, source=str(path))
            except StateValidationError as exc:
                result.update(valid_json=False, error=str(exc))
                summaries.append(result)
                continue
            result.update(valid_json=True, schema_version=payload["schema_version"])
            if path == self.standard_data_path:
                baseline = payload["baseline"]
                assert isinstance(baseline, dict)
                series = baseline["series"]
                assert isinstance(series, list)
                result.update(
                    start_step=baseline["start_step"],
                    end_step=baseline["end_step"],
                    series_count=len(series),
                )
            else:
                events = payload["events"]
                assert isinstance(events, list)
                result["event_count"] = len(events)
            summaries.append(result)
        return summaries

    def validate_state_targets(self) -> tuple[Path, ...]:
        """Return existing regular state files that a safe reset may touch."""

        for path in self.state_paths:
            _validate_parent_directory(path.parent)
        targets = []
        for path in self.state_paths:
            if not _path_present(path):
                continue
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                raise StorageError(f"cannot inspect {path}: {exc}") from exc
            if not stat.S_ISREG(mode):
                raise StorageError(f"refusing to reset non-regular state path: {path}")
            targets.append(path)
        return tuple(targets)

    def backup_state(
        self, backup_dir: Path, targets: Sequence[Path]
    ) -> tuple[Path, ...]:
        """Copy every validated target into a new backup directory."""

        checked = self._validate_requested_targets(targets)
        try:
            Path(backup_dir).mkdir(parents=True, exist_ok=False)
            backups = []
            for source in checked:
                destination = Path(backup_dir) / source.name
                shutil.copy2(source, destination, follow_symlinks=False)
                backups.append(destination)
        except OSError as exc:
            raise StorageError(f"cannot back up degradation state: {exc}") from exc
        return tuple(backups)

    def delete_state(self, targets: Sequence[Path]) -> tuple[Path, ...]:
        """Delete only configured targets after validating all of them."""

        checked = self._validate_requested_targets(targets)
        deleted = []
        for path in checked:
            try:
                path.unlink()
            except OSError as exc:
                raise StorageError(f"cannot delete {path}: {exc}") from exc
            deleted.append(path)
        return tuple(deleted)

    def _validate_requested_targets(self, targets: Sequence[Path]) -> tuple[Path, ...]:
        allowed = set(self.state_paths)
        requested = tuple(Path(path) for path in targets)
        if len(set(requested)) != len(requested) or any(
            path not in allowed for path in requested
        ):
            raise StorageError("state targets must be configured degradation files")
        existing = set(self.validate_state_targets())
        if any(path not in existing for path in requested):
            raise StorageError("a requested degradation state file is unavailable")
        return requested

    def baseline_exists(self) -> bool:
        """Return whether a frozen baseline file exists."""

        return self.standard_data_path.exists()

    def load_baseline(self) -> BaselineSnapshot:
        """Load a frozen baseline from standard_data.json."""

        return _baseline_from_json(_read_json(self.standard_data_path))

    def save_baseline(self, snapshot: BaselineSnapshot) -> None:
        """Save a frozen baseline without raw training observations."""

        if not isinstance(snapshot, BaselineSnapshot):
            raise TypeError("snapshot must be BaselineSnapshot")
        _write_json(self.standard_data_path, _baseline_to_json(snapshot))

    def save_event(
        self,
        phase: str,
        event: DegradationEvent,
        result: AssociationResult,
        top_associations: Sequence[AssociationItem],
        directions: Mapping[SeriesId, str],
    ) -> None:
        """Create a confirmed event or update it when the event closes."""

        if phase not in {"confirmed", "closed"}:
            raise ValueError("phase must be confirmed or closed")
        payload = _load_abnormal_data(self.abnormal_data_path)
        events = payload["events"]
        association = _association_to_json(result, top_associations, directions)
        if phase == "confirmed":
            events.append(
                {
                    "target": event.identity.name,
                    "target_labels": dict(event.identity.labels),
                    "direction": event.direction.value,
                    "start_step": event.start_step,
                    "start_time": event.start_time,
                    "end_step": event.end_step,
                    "end_time": event.end_time,
                    "confirmed_at_step": event.confirmed_at_step,
                    "confirmed_at_time": event.confirmed_at_time,
                    "closed_at_step": None,
                    "closed_at_time": None,
                    "abnormal_points": event.abnormal_points,
                    "association": {"confirmed": association, "closed": None},
                }
            )
        else:
            matching = [
                item
                for item in events
                if item.get("target") == event.identity.name
                and item.get("target_labels") == dict(event.identity.labels)
                and item.get("start_step") == event.start_step
                and item.get("confirmed_at_step") == event.confirmed_at_step
            ]
            if len(matching) != 1:
                raise StorageError("cannot find the confirmed event to close")
            record = matching[0]
            record.update(
                {
                    "end_step": event.end_step,
                    "end_time": event.end_time,
                    "closed_at_step": event.closed_at_step,
                    "closed_at_time": event.closed_at_time,
                    "abnormal_points": event.abnormal_points,
                }
            )
            stored_association = record.get("association")
            if not isinstance(stored_association, dict):
                raise StorageError("abnormal_data.json is invalid")
            stored_association["closed"] = association
        _write_json(self.abnormal_data_path, payload)

    def reset(self) -> None:
        """Delete only the two configured degradation state files."""

        for path in (self.standard_data_path, self.abnormal_data_path):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                raise StorageError(f"cannot delete {path}: {exc}") from exc


__all__ = [
    "ABNORMAL_DATA_NAME",
    "STANDARD_DATA_NAME",
    "BaselineSnapshot",
    "JsonStorage",
    "StorageError",
]
