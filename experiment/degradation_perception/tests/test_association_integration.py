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
import importlib.util

import pytest

from experiment.degradation_perception.algorithm import DegradationPerception
from experiment.degradation_perception.perception_config import TimeSeries
from experiment.degradation_perception.policy import resolve_metric_policy


TARGET = "timing_s/step"
STRONG = "gpu_utilization"
WEAKER = "memory_usage"
NORMAL = "network_latency"
STANDARD_PATTERN = [
    1.00,
    1.01,
    0.99,
    1.02,
    1.00,
    0.98,
    1.01,
    1.00,
    0.99,
    1.02,
] * 4


def _standard(level: float) -> list[float]:
    return [level * value for value in STANDARD_PATTERN]


def _inference(
    level: float,
    events: list[tuple[int, int, float]],
    *,
    length: int = 60,
) -> list[float]:
    values = [level * (1.0 + (index % 5 - 2) * 0.003) for index in range(length)]
    for start, stop, event_level in events:
        for index in range(start, stop + 1):
            values[index] = event_level + (index - start) * 0.01
    return values


def _dataset(
    specifications: dict[
        str,
        tuple[float, list[tuple[int, int, float]]],
    ],
) -> dict:
    return {
        "standard": {
            metric: TimeSeries(
                timestamps=list(range(1, len(STANDARD_PATTERN) + 1)),
                values=_standard(level),
            )
            for metric, (level, _) in specifications.items()
        },
        "inference": {
            metric: TimeSeries(
                timestamps=list(range(100, 160)),
                values=_inference(level, events),
            )
            for metric, (level, events) in specifications.items()
        },
    }


def _enable_association(
    *,
    target_metrics=None,
    context_ratio=0.5,
    min_aligned_points=10,
    min_rf_samples=20,
    top_k=5,
) -> dict:
    targets = list(target_metrics or [TARGET])
    data = resolve_metric_policy(targets[0])
    data["association"].update(
        {
            "enabled": True,
            "target_metrics": targets,
            "context_ratio": context_ratio,
            "min_aligned_points": min_aligned_points,
            "min_rf_samples": min_rf_samples,
            "min_coverage_ratio": 0.8,
            "top_k": top_k,
        }
    )
    data["association"]["random_forest"]["n_estimators"] = 64
    return {targets[0]: data}


def _primary_dataset() -> dict:
    return _dataset(
        {
            TARGET: (1.0, [(20, 34, 1.50)]),
            STRONG: (10.0, [(20, 34, 20.0)]),
            WEAKER: (5.0, [(25, 34, 9.0)]),
            NORMAL: (2.0, []),
        }
    )


def test_disabled_association_preserves_original_top_level_shape():
    response = DegradationPerception(
        dataset=_primary_dataset(),
        metrics=[TARGET, STRONG, WEAKER, NORMAL],
    ).detect()
    assert set(response) == {
        "taskId",
        "states",
        "results",
        "abnormalTimeRange",
    }
    assert "associationAnalysis" not in response


def test_real_kde_then_association_produces_ranked_event_and_rf_evidence():
    metric_policies = _enable_association()
    response = DegradationPerception(
        dataset=_primary_dataset(),
        metrics=[TARGET, STRONG, WEAKER, NORMAL],
        metric_policies=metric_policies,
    ).detect()
    assert response["states"] == {
        TARGET: 0,
        STRONG: 0,
        WEAKER: 0,
        NORMAL: 0,
    }
    assert response["abnormalTimeRange"][TARGET]
    analysis = response["associationAnalysis"]
    target = analysis["targets"][TARGET]
    assert target["status"] in {"success", "partial_success"}
    assert len(target["events"]) == 1
    event = target["events"][0]
    assert event["targetAbnormalRange"] == response["abnormalTimeRange"][TARGET][0]
    assert event["rawTargetAbnormalRange"] == {
        "startTime": 120.0,
        "endTime": 134.0,
    }
    assert event["analysisWindow"] == {
        "startTime": 113.0,
        "endTime": 141.0,
        "usedMinimumContext": False,
    }
    assert event["topAssociations"][0]["metric"] == STRONG
    assert event["topAssociations"][0]["rank"] == 1
    if importlib.util.find_spec("sklearn") is None:
        assert event["randomForestStatus"] == "dependency_unavailable"
        assert event["randomForestDiagnostics"]["importanceMethod"] is None
    else:
        assert event["randomForestStatus"] in {"success", "partial_success"}
        assert event["randomForestDiagnostics"]["importanceMethod"] in {
            "permutation",
            "impurity_fallback",
        }
    assert any(
        item
        == {
            "metric": NORMAL,
            "reason": "not_abnormal_in_target_window",
        }
        for item in event["excludedMetrics"]
    )
    json.dumps(response, allow_nan=False)


def test_missing_or_normal_target_returns_business_status_not_exception(tmp_path):
    missing = DegradationPerception(
        dataset=_primary_dataset(),
        metrics=[STRONG],
        association_targets=[TARGET],
    ).detect()
    assert missing["associationAnalysis"]["targets"][TARGET]["status"] == (
        "target_metric_missing"
    )

    selected_but_missing_data = _primary_dataset()
    selected_but_missing_data["inference"].pop(TARGET)
    selected_but_missing = DegradationPerception(
        dataset=selected_but_missing_data,
        metrics=[TARGET, STRONG],
        association_targets=[TARGET],
    ).detect()
    assert (
        selected_but_missing["associationAnalysis"]["targets"][TARGET]["status"]
        == "target_metric_missing"
    )

    normal_data = _primary_dataset()
    target_series = normal_data["inference"][TARGET]
    normal_data["inference"][TARGET] = TimeSeries(
        timestamps=list(target_series.timestamps),
        values=_inference(1.0, []),
    )
    normal = DegradationPerception(
        dataset=normal_data,
        metrics=[TARGET, STRONG],
        association_targets=[TARGET],
    ).detect()
    assert normal["associationAnalysis"]["targets"][TARGET]["status"] == (
        "target_not_abnormal"
    )


def test_failed_lower_metric_is_excluded_without_changing_target_kde(tmp_path):
    data = _primary_dataset()
    data["standard"]["broken_metric"] = TimeSeries(
        timestamps=[1, 2, 3],
        values=[1.0],
    )
    data["inference"]["broken_metric"] = TimeSeries(
        timestamps=list(range(100, 160)),
        values=[1.0] * 60,
    )
    response = DegradationPerception(
        dataset=data,
        metrics=[TARGET, STRONG, "broken_metric"],
        association_targets=[TARGET],
    ).detect()
    assert response["states"][TARGET] == 0
    assert response["abnormalTimeRange"][TARGET]
    event = response["associationAnalysis"]["targets"][TARGET]["events"][0]
    assert {
        "metric": "broken_metric",
        "reason": "candidate_detection_failed",
    } in event["excludedMetrics"]


def test_constant_candidate_is_identified_before_abnormal_window_filter(tmp_path):
    data = _primary_dataset()
    data["standard"]["constant_metric"] = TimeSeries(
        timestamps=list(range(1, len(STANDARD_PATTERN) + 1)),
        values=[5.0] * len(STANDARD_PATTERN),
    )
    data["inference"]["constant_metric"] = TimeSeries(
        timestamps=list(range(100, 160)),
        values=[5.0] * 60,
    )
    response = DegradationPerception(
        dataset=data,
        metrics=[TARGET, STRONG, "constant_metric"],
        association_targets=[TARGET],
    ).detect()

    event = response["associationAnalysis"]["targets"][TARGET]["events"][0]
    assert {
        "metric": "constant_metric",
        "reason": "constant_candidate_series",
    } in event["excludedMetrics"]


def test_multiple_target_events_are_analyzed_independently(tmp_path):
    first = "first_metric"
    second = "second_metric"
    data = _dataset(
        {
            TARGET: (1.0, [(10, 17, 1.5), (40, 47, 1.6)]),
            first: (5.0, [(10, 17, 9.0)]),
            second: (8.0, [(40, 47, 14.0)]),
        }
    )
    metric_policies = _enable_association(
        context_ratio=0.5,
        min_aligned_points=5,
        min_rf_samples=100,
    )
    response = DegradationPerception(
        dataset=data,
        metrics=[TARGET, first, second],
        metric_policies=metric_policies,
    ).detect()
    events = response["associationAnalysis"]["targets"][TARGET]["events"]
    assert len(events) == 2
    assert [event["topAssociations"][0]["metric"] for event in events] == [
        first,
        second,
    ]
    assert {
        "metric": second,
        "reason": "not_abnormal_in_target_window",
    } in events[0]["excludedMetrics"]
    assert {
        "metric": first,
        "reason": "not_abnormal_in_target_window",
    } in events[1]["excludedMetrics"]


def test_top_five_truncation_keeps_full_contribution_and_stable_ties(tmp_path):
    candidates = [f"candidate_{letter}" for letter in "fedcba"]
    specifications = {TARGET: (1.0, [(20, 34, 1.5)])}
    specifications.update(
        {
            metric: (float(index + 2), [(20, 34, float(index + 5))])
            for index, metric in enumerate(candidates)
        }
    )
    metric_policies = _enable_association(min_rf_samples=100)
    response = DegradationPerception(
        dataset=_dataset(specifications),
        metrics=[TARGET, *candidates],
        metric_policies=metric_policies,
    ).detect()
    event = response["associationAnalysis"]["targets"][TARGET]["events"][0]
    assert len(event["topAssociations"]) == 5
    assert len(event["allAssociations"]) == 6
    assert [item["rank"] for item in event["topAssociations"]] == [1, 2, 3, 4, 5]
    assert [item["metric"] for item in event["topAssociations"]] == sorted(candidates)[
        :5
    ]
    assert sum(
        item["abnormalContribution"] for item in event["allAssociations"]
    ) == pytest.approx(100.0)
    assert (
        sum(item["abnormalContribution"] for item in event["topAssociations"]) < 100.0
    )


@pytest.mark.parametrize("candidate_count", [3, 5, 8])
def test_top_five_handles_candidate_counts_below_equal_and_above_limit(
    tmp_path,
    candidate_count,
):
    candidates = [f"candidate_{index}" for index in range(candidate_count)]
    specifications = {TARGET: (1.0, [(20, 34, 1.5)])}
    specifications.update(
        {
            metric: (float(index + 2), [(20, 34, float(index + 5))])
            for index, metric in enumerate(candidates)
        }
    )
    metric_policies = _enable_association(
        min_rf_samples=100,
        top_k=5,
    )

    response = DegradationPerception(
        dataset=_dataset(specifications),
        metrics=[TARGET, *candidates],
        metric_policies=metric_policies,
    ).detect()
    event = response["associationAnalysis"]["targets"][TARGET]["events"][0]
    all_associations = event["allAssociations"]
    top_associations = event["topAssociations"]

    assert len(all_associations) == candidate_count
    assert len(top_associations) == min(5, candidate_count)
    assert top_associations == all_associations[:5]
    assert sum(
        item["abnormalContribution"] for item in all_associations
    ) == pytest.approx(100.0)
    top_total = sum(
        item["abnormalContribution"] for item in top_associations
    )
    if candidate_count <= 5:
        assert top_total == pytest.approx(100.0)
    else:
        assert top_total < 100.0


def test_top_k_changes_only_the_view_not_full_contributions(tmp_path):
    candidates = [f"candidate_{index}" for index in range(8)]
    specifications = {TARGET: (1.0, [(20, 34, 1.5)])}
    specifications.update(
        {
            metric: (float(index + 2), [(20, 34, float(index + 5))])
            for index, metric in enumerate(candidates)
        }
    )
    dataset = _dataset(specifications)
    contributions_by_top_k = {}

    for top_k in (3, 5, 8):
        metric_policies = _enable_association(
            min_rf_samples=100,
            top_k=top_k,
        )
        response = DegradationPerception(
            dataset=dataset,
            metrics=[TARGET, *candidates],
            metric_policies=metric_policies,
        ).detect()
        event = response["associationAnalysis"]["targets"][TARGET]["events"][0]
        all_associations = event["allAssociations"]
        top_associations = event["topAssociations"]
        contributions_by_top_k[top_k] = {
            item["metric"]: item["abnormalContribution"]
            for item in all_associations
        }

        assert len(all_associations) == 8
        assert len(top_associations) == top_k
        assert top_associations == all_associations[:top_k]
        assert sum(
            item["abnormalContribution"] for item in all_associations
        ) == pytest.approx(100.0)
        top_total = sum(
            item["abnormalContribution"] for item in top_associations
        )
        if top_k == 8:
            assert top_total == pytest.approx(100.0)
        else:
            assert top_total < 100.0

    baseline = contributions_by_top_k[3]
    for top_k in (5, 8):
        assert contributions_by_top_k[top_k] == pytest.approx(baseline)


def test_multiple_top_metrics_are_isolated_and_never_rank_each_other(tmp_path):
    second_target = "request_latency"
    candidate = "cpu_usage"
    data = _dataset(
        {
            TARGET: (1.0, [(20, 34, 1.5)]),
            second_target: (2.0, [(20, 34, 3.5)]),
            candidate: (10.0, [(20, 34, 20.0)]),
        }
    )
    response = DegradationPerception(
        dataset=data,
        metrics=[TARGET, second_target, candidate],
        association_targets=[TARGET, second_target],
    ).detect()
    targets = response["associationAnalysis"]["targets"]
    assert set(targets) == {TARGET, second_target}
    for result in targets.values():
        ranked = {
            item["metric"]
            for event in result["events"]
            for item in event["allAssociations"]
        }
        assert ranked == {candidate}
    first_event = targets[TARGET]["events"][0]
    second_event = targets[second_target]["events"][0]
    assert {
        "metric": second_target,
        "reason": "configured_target_metric",
    } in first_event["excludedMetrics"]
    assert {"metric": TARGET, "reason": "configured_target_metric"} in (
        second_event["excludedMetrics"]
    )
