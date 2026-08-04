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

"""Shared runtime helpers for local and SSH-polled training logs.

This module is deliberately an adapter around :class:`DegradationPerception`.
It does not implement KDE fitting, stable-segment discovery, point
classification, or abnormal-interval merging.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .algorithm import DegradationPerception
from .perception_config import DetectionInput, DetectionResult, TimeSeries
from .policy import resolve_metric_policy
from .serialization import to_json_serializable
from .timeseries import validate_detection_input


def _metric_names(metrics: Sequence[str] | str) -> list[str]:
    selected = [metrics] if isinstance(metrics, str) else list(metrics)
    selected = list(dict.fromkeys(str(metric) for metric in selected))
    if not selected or any(not metric.strip() for metric in selected):
        raise ValueError("metrics must contain at least one non-empty name")
    return selected


def build_detection_dataset(
    standard: Mapping[str, TimeSeries],
    inference: Mapping[str, TimeSeries],
    metrics: Sequence[str] | str,
) -> DetectionInput:
    """Build and validate two independent algorithm phases.

    Missing metrics become empty series in their own phase. Values are never
    copied between ``standard`` and ``inference``.
    """

    selected = _metric_names(metrics)
    payload: DetectionInput = {
        "standard": {
            metric: standard.get(metric, TimeSeries()) for metric in selected
        },
        "inference": {
            metric: inference.get(metric, TimeSeries()) for metric in selected
        },
    }
    return validate_detection_input(payload)


class DetectionRunner:
    """Run repeated detections against one immutable healthy baseline."""

    def __init__(
        self,
        *,
        standard: Mapping[str, TimeSeries],
        metrics: Sequence[str] | str,
        task_id: str | None = None,
        source_type: str = "training_log",
        metric_policies: Mapping[str, Mapping[str, Any]] | None = None,
        history_policy: Mapping[str, Any] | None = None,
        association_targets: Sequence[str] | str | None = None,
    ) -> None:
        self.metrics = _metric_names(metrics)
        canonical = build_detection_dataset(standard, {}, self.metrics)
        self.standard = canonical["standard"]
        self.metric_policies = dict(metric_policies or {})
        self.detector = DegradationPerception(
            metrics=self.metrics,
            task_id=task_id,
            source_type=source_type,
            metric_policies=self.metric_policies,
            history_policy=history_policy,
            association_targets=association_targets,
        )

    def minimum_inference_samples(self) -> dict[str, int]:
        """Return the effective per-metric minimum used by the algorithm."""

        return {
            metric: int(
                resolve_metric_policy(
                    metric,
                    self.metric_policies.get(metric),
                )["minimum_inference_points"]
            )
            for metric in self.metrics
        }

    def run(self, inference: Mapping[str, TimeSeries]) -> DetectionResult:
        """Detect one inference mapping and attach observation-only KDE data."""

        dataset = build_detection_dataset(self.standard, inference, self.metrics)
        result = self.detector.detect_dataset(dataset, metrics=self.metrics)
        result["debugDetails"] = {  # type: ignore[typeddict-unknown-key]
            "metrics": build_kde_debug_details(
                self.detector,
                result,
                dataset,
                self.metrics,
            )
        }
        return to_json_serializable(result)


def _model_mapping(model: Any) -> dict[str, Any]:
    converted = to_json_serializable(model)
    if not isinstance(converted, Mapping):
        raise TypeError("threshold model must serialize to an object")
    return dict(converted)


def build_kde_debug_details(
    detector: DegradationPerception,
    result: Mapping[str, Any],
    dataset: DetectionInput,
    metrics: Sequence[str] | str,
) -> dict[str, dict[str, Any]]:
    """Summarize existing model fields without recalculating thresholds."""

    selected = _metric_names(metrics)
    canonical = validate_detection_input(dataset)
    result_details = result.get("results", {})
    states = result.get("states", {})
    published = result.get("abnormalTimeRange", {})
    summaries: dict[str, dict[str, Any]] = {}

    for metric in selected:
        config = detector.config_dict.get(metric)
        if config is None:
            try:
                config = resolve_metric_policy(
                    metric,
                    detector.metric_policies.get(metric),
                )
            except Exception:
                config = {}

        metric_result = (
            result_details.get(metric, {})
            if isinstance(result_details, Mapping)
            else {}
        )
        if not isinstance(metric_result, Mapping):
            metric_result = {}
        raw_models = metric_result.get("thresholds", [])
        if not isinstance(raw_models, list):
            raw_models = []
        if not raw_models:
            raw_models = list(detector.standard_models.get(metric, []))

        lower_ratio = config.get("lower_ratio")
        upper_ratio = config.get("upper_ratio")
        model_summaries: list[dict[str, Any]] = []
        for raw_model in raw_models:
            model = _model_mapping(raw_model)
            diagnostics = model.get("diagnostics", {})
            if not isinstance(diagnostics, Mapping):
                diagnostics = {}
            model_summaries.append(
                {
                    "modelId": model.get("mode_id"),
                    "stableSegmentStepRange": {
                        "startStep": model.get("segment_start_time"),
                        "endStep": model.get("segment_end_time"),
                    },
                    "standardSegmentSamples": model.get("point_count"),
                    "rawLowerThreshold": model.get("lower_kde_threshold"),
                    "rawUpperThreshold": model.get("upper_kde_threshold"),
                    "finalLowerThreshold": model.get("lower_threshold"),
                    "finalUpperThreshold": model.get("upper_threshold"),
                    "lowerRatio": lower_ratio,
                    "upperRatio": upper_ratio,
                    "effectiveBandwidth": model.get("bandwidth"),
                    "peakInfluenceRegion": diagnostics.get("influenceRegion"),
                }
            )

        point_diagnostics = metric_result.get("pointDiagnostics")
        state = states.get(metric) if isinstance(states, Mapping) else None
        abnormal_points: int | None = None
        if state == 0 and isinstance(point_diagnostics, list):
            abnormal_points = sum(
                bool(item.get("abnormal"))
                for item in point_diagnostics
                if isinstance(item, Mapping)
            )
        current_ranges = metric_result.get("currentAbnormalTimeRange", [])
        if not isinstance(current_ranges, list):
            current_ranges = []
        published_ranges = (
            published.get(metric, []) if isinstance(published, Mapping) else []
        )
        if not isinstance(published_ranges, list):
            published_ranges = []

        alpha = config.get("alpha")
        lower_probability = float(alpha) if alpha is not None else None
        upper_probability = (
            1.0 - lower_probability if lower_probability is not None else None
        )
        modeled_samples = sum(
            int(model.get("standardSegmentSamples") or 0)
            for model in model_summaries
        )
        standard_samples = len(canonical["standard"].get(metric, TimeSeries()))
        inference_samples = len(canonical["inference"].get(metric, TimeSeries()))
        summaries[metric] = {
            "state": state,
            "standardSamples": standard_samples,
            "inferenceSamples": inference_samples,
            "alphaParameterName": "alpha",
            "alpha": lower_probability,
            "lowerProbability": lower_probability,
            "upperProbability": upper_probability,
            "lowerRatio": lower_ratio,
            "upperRatio": upper_ratio,
            "configuredBandwidth": (
                config.get("kde", {}).get("bandwidth")
                if isinstance(config.get("kde"), Mapping)
                else None
            ),
            "normalModelCount": len(model_summaries),
            "modeledStandardSamples": modeled_samples,
            "normalModels": model_summaries,
            "abnormalPointCount": abnormal_points,
            "currentAbnormalIntervalCount": len(current_ranges),
            "abnormalIntervalCount": len(published_ranges),
        }
    return to_json_serializable(summaries)


def _display_number(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _display_event_time(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)
    return str(int(number)) if number.is_integer() else str(number)


def format_terminal_summary(
    result: Mapping[str, Any],
    *,
    debug_kde: bool = False,
) -> str:
    """Format a human-readable summary; rounding is display-only."""

    lines = [f"Task: {result.get('taskId', 'default')}"]
    debug_root = result.get("debugDetails", {})
    debug_metrics = (
        debug_root.get("metrics", {}) if isinstance(debug_root, Mapping) else {}
    )
    states = result.get("states", {})
    for metric, details in (
        debug_metrics.items() if isinstance(debug_metrics, Mapping) else []
    ):
        if not isinstance(details, Mapping):
            continue
        lines.extend(
            [
                "",
                f"Metric: {metric}",
                f"State: {states.get(metric, details.get('state')) if isinstance(states, Mapping) else details.get('state')}",
                f"Abnormal intervals: {details.get('abnormalIntervalCount', 0)}",
            ]
        )
        if not debug_kde:
            continue
        lines.extend(
            [
                f"Standard samples: {details.get('standardSamples', 0)}",
                f"Inference samples: {details.get('inferenceSamples', 0)}",
                "",
                f"Alpha: {_display_number(details.get('alpha'))}",
                "Lower probability: "
                f"{_display_number(details.get('lowerProbability'))}",
                "Upper probability: "
                f"{_display_number(details.get('upperProbability'))}",
                f"Normal models: {details.get('normalModelCount', 0)}",
            ]
        )
        models = details.get("normalModels", [])
        if isinstance(models, list):
            for index, model in enumerate(models, start=1):
                if not isinstance(model, Mapping):
                    continue
                lines.extend(
                    [
                        "",
                        f"Model {index}",
                        "  KDE standard range: "
                        f"{_display_number(model.get('finalLowerThreshold'))} ~ "
                        f"{_display_number(model.get('finalUpperThreshold'))}",
                        "  Raw KDE range: "
                        f"{_display_number(model.get('rawLowerThreshold'))} ~ "
                        f"{_display_number(model.get('rawUpperThreshold'))}",
                        "  Lower ratio: "
                        f"{_display_number(model.get('lowerRatio'))}",
                        "  Upper ratio: "
                        f"{_display_number(model.get('upperRatio'))}",
                        "  KDE bandwidth: "
                        f"{_display_number(model.get('effectiveBandwidth'))}",
                    ]
                )
        lines.extend(
            [
                "",
                "Detection",
                "  Abnormal points: "
                f"{details.get('abnormalPointCount', 'N/A')}",
                "  Abnormal intervals: "
                f"{details.get('abnormalIntervalCount', 0)}",
            ]
        )
    if "abnormalMetrics" in result:
        abnormal_metrics = result.get("abnormalMetrics")
        if isinstance(abnormal_metrics, list) and abnormal_metrics:
            lines.extend(["", "Abnormal metrics:", ""])
            for metric_entry in abnormal_metrics:
                if not isinstance(metric_entry, Mapping):
                    continue
                lines.append(str(metric_entry.get("metric", "")))
                events = metric_entry.get("events", [])
                if not isinstance(events, list):
                    continue
                for event in events:
                    if not isinstance(event, Mapping):
                        continue
                    lines.append(
                        f"  event {event.get('eventIndex')}: "
                        f"{_display_event_time(event.get('startTime'))} -> "
                        f"{_display_event_time(event.get('endTime'))}"
                    )
        else:
            lines.extend(["", "No abnormal metrics detected."])
    return "\n".join(lines) + "\n"


def serialize_result(result: Mapping[str, Any]) -> str:
    """Serialize with full float precision and reject JSON NaN extensions."""

    return json.dumps(
        to_json_serializable(result),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    )


def save_result(result: Mapping[str, Any], output: str | Path) -> Path:
    """Save one standards-compliant UTF-8 JSON result."""

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialize_result(result) + "\n", encoding="utf-8")
    return destination


__all__ = [
    "DetectionRunner",
    "build_detection_dataset",
    "build_kde_debug_details",
    "format_terminal_summary",
    "save_result",
    "serialize_result",
]
