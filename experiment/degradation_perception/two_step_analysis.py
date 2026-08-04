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

"""Adapters for selecting and analyzing one previously detected event."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .association_analysis import AssociationAnalyzer
from .perception_config import TimeSeries
from .serialization import to_json_serializable


def build_abnormal_metrics(
    detection_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Summarize published KDE intervals without re-evaluating anomalies."""

    published = detection_result.get("abnormalTimeRange", {})
    if not isinstance(published, Mapping):
        raise ValueError("detection result abnormalTimeRange must be an object")
    abnormal_metrics: list[dict[str, Any]] = []
    for metric, raw_events in published.items():
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError("detection result contains an invalid metric name")
        if not isinstance(raw_events, Sequence) or isinstance(
            raw_events, (str, bytes)
        ):
            raise ValueError(f"abnormal intervals for {metric!r} must be a list")
        events: list[dict[str, Any]] = []
        for event_index, raw_event in enumerate(raw_events):
            if not isinstance(raw_event, Mapping):
                raise ValueError(
                    f"abnormal event {event_index} for {metric!r} must be an object"
                )
            if "startTime" not in raw_event or "endTime" not in raw_event:
                raise ValueError(
                    f"abnormal event {event_index} for {metric!r} is missing bounds"
                )
            events.append(
                {
                    "eventIndex": event_index,
                    "startTime": copy.deepcopy(raw_event["startTime"]),
                    "endTime": copy.deepcopy(raw_event["endTime"]),
                }
            )
        if events:
            abnormal_metrics.append({"metric": metric, "events": events})
    return to_json_serializable(abnormal_metrics)


def attach_abnormal_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Attach the two-step event index to one mutable detection result."""

    result["abnormalMetrics"] = build_abnormal_metrics(result)
    return result


def _finite_event_bound(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"selected abnormal event {name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"selected abnormal event {name} must be a finite number")
    return number


def analyze_detection_event(
    detection_result: Mapping[str, Any],
    inference: Mapping[str, TimeSeries],
    *,
    target_metric: str,
    event_index: int,
    top_k: int,
    association_config: Mapping[str, Any],
    source_type: str = "training_log",
) -> dict[str, Any]:
    """Run the existing association analyzer for one saved KDE event."""

    if not isinstance(target_metric, str) or not target_metric.strip():
        raise ValueError("target metric must be a non-empty string")
    if isinstance(event_index, bool) or not isinstance(event_index, int):
        raise ValueError("event-index must be a non-negative integer")
    if event_index < 0:
        raise ValueError("event-index must be a non-negative integer")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top-k must be a positive integer")
    if not isinstance(detection_result, Mapping):
        raise ValueError("detection result root must be an object")
    if not isinstance(inference, Mapping):
        raise ValueError("inference data must be keyed by metric")

    states = detection_result.get("states", {})
    results = detection_result.get("results", {})
    published = detection_result.get("abnormalTimeRange", {})
    if not all(isinstance(item, Mapping) for item in (states, results, published)):
        raise ValueError("detection result has an invalid metric structure")
    known_metrics = set(states) | set(results) | set(published)
    if target_metric not in known_metrics:
        raise ValueError(
            f"target metric {target_metric!r} is absent from the detection result"
        )

    raw_events = published.get(target_metric, [])
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)):
        raise ValueError(
            f"abnormal intervals for target metric {target_metric!r} must be a list"
        )
    if not raw_events:
        raise ValueError(f"target metric {target_metric!r} has no abnormal events")
    if event_index >= len(raw_events):
        raise ValueError(
            f"event-index {event_index} is out of range for target metric "
            f"{target_metric!r}; available events: {len(raw_events)}"
        )
    selected_event = raw_events[event_index]
    if not isinstance(selected_event, Mapping):
        raise ValueError("selected abnormal event must be an object")
    raw_start = _finite_event_bound(selected_event.get("startTime"), "startTime")
    raw_end = _finite_event_bound(selected_event.get("endTime"), "endTime")
    if raw_start > raw_end:
        raise ValueError("selected abnormal event startTime must not exceed endTime")
    if target_metric not in inference:
        raise ValueError(
            f"target metric {target_metric!r} is absent from the inference log"
        )

    candidate_metrics = list(
        dict.fromkeys(
            [
                *states,
                *(
                    detection_result.get("metricErrors", {})
                    if isinstance(detection_result.get("metricErrors", {}), Mapping)
                    else {}
                ),
            ]
        )
    )
    metric_context: dict[str, dict[str, Any]] = {
        metric: {
            "input_present": metric in inference,
            "series": inference.get(metric),
            "raw_events": [],
        }
        for metric in candidate_metrics
    }
    metric_context.setdefault(
        target_metric,
        {
            "input_present": True,
            "series": inference[target_metric],
            "raw_events": [],
        },
    )
    metric_context[target_metric]["raw_events"] = [
        {
            "rawStartTime": raw_start,
            "rawEndTime": raw_end,
            "targetAbnormalRange": copy.deepcopy(dict(selected_event)),
        }
    ]

    selected_output = copy.deepcopy(dict(detection_result))
    selected_ranges = copy.deepcopy(dict(published))
    selected_ranges[target_metric] = [copy.deepcopy(dict(selected_event))]
    selected_output["abnormalTimeRange"] = selected_ranges

    config = copy.deepcopy(dict(association_config))
    config["enabled"] = True
    config["target_metrics"] = [target_metric]
    config["top_k"] = top_k
    analysis = AssociationAnalyzer(config, source_type=source_type).analyze(
        metric_context,
        selected_output,
    )
    target_analysis = analysis.get("targets", {}).get(target_metric, {})
    analyzed_events = target_analysis.get("events", [])
    if not isinstance(analyzed_events, list) or not analyzed_events:
        reason = target_analysis.get("reason", target_analysis.get("status"))
        raise ValueError(
            f"association analysis could not analyze the selected event: {reason}"
        )
    event_analysis = analyzed_events[0]
    if not isinstance(event_analysis, Mapping):
        raise ValueError("association analysis returned an invalid event")

    output = {
        "targetMetric": target_metric,
        "eventIndex": event_index,
        "abnormalRange": {
            "startTime": copy.deepcopy(selected_event["startTime"]),
            "endTime": copy.deepcopy(selected_event["endTime"]),
        },
        "status": event_analysis.get("status"),
        "analysisWindow": copy.deepcopy(event_analysis.get("analysisWindow")),
        "candidateMetricCount": event_analysis.get("candidateMetricCount", 0),
        "validCandidateMetricCount": event_analysis.get(
            "validCandidateMetricCount", 0
        ),
        "excludedMetrics": copy.deepcopy(event_analysis.get("excludedMetrics", [])),
        "candidateDiagnostics": copy.deepcopy(
            event_analysis.get("candidateDiagnostics", {})
        ),
        "alignedSampleCount": event_analysis.get("alignedSampleCount", 0),
        "randomForestStatus": event_analysis.get("randomForestStatus"),
        "randomForestDiagnostics": copy.deepcopy(
            event_analysis.get("randomForestDiagnostics", {})
        ),
        "weights": copy.deepcopy(analysis.get("weights", {})),
        "topAssociations": copy.deepcopy(
            event_analysis.get("topAssociations", [])
        ),
    }
    return to_json_serializable(output)


__all__ = [
    "analyze_detection_event",
    "attach_abnormal_metrics",
    "build_abnormal_metrics",
]
