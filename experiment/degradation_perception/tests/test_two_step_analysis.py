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
from pathlib import Path

import pytest

from experiment.degradation_perception import main as cli
from experiment.degradation_perception.association_analysis import (
    DEFAULT_ASSOCIATION_CONFIG,
)
from experiment.degradation_perception.detection_runtime import (
    format_terminal_summary,
    serialize_result,
)
from experiment.degradation_perception.perception_config import TimeSeries
from experiment.degradation_perception.two_step_analysis import (
    analyze_detection_event,
    attach_abnormal_metrics,
    build_abnormal_metrics,
)


TARGET = "timing_s/step"
CANDIDATES = ["gpu_utilization", "hbm_bandwidth"]
TIMESTAMPS = list(range(20))


def _diagnostics(abnormal_steps: set[int]) -> list[dict[str, object]]:
    return [
        {"timestamp": step, "abnormal": step in abnormal_steps}
        for step in TIMESTAMPS
    ]


def _detection_result(
    events: list[dict[str, float]] | None = None,
) -> dict[str, object]:
    selected_events = (
        [{"startTime": 8.0, "endTime": 11.0}]
        if events is None
        else events
    )
    metrics = [TARGET, *CANDIDATES]
    return {
        "taskId": "two-step-test",
        "states": {metric: 0 for metric in metrics},
        "results": {
            TARGET: {"pointDiagnostics": _diagnostics({8, 9, 10, 11})},
            CANDIDATES[0]: {"pointDiagnostics": _diagnostics({9, 10})},
            CANDIDATES[1]: {"pointDiagnostics": _diagnostics({10, 11})},
        },
        "abnormalTimeRange": {
            TARGET: selected_events,
            CANDIDATES[0]: [],
            CANDIDATES[1]: [],
        },
    }


def _inference() -> dict[str, TimeSeries]:
    return {
        TARGET: TimeSeries(TIMESTAMPS, [float(step) for step in TIMESTAMPS]),
        CANDIDATES[0]: TimeSeries(
            TIMESTAMPS,
            [float(step * 2 + (step % 2)) for step in TIMESTAMPS],
        ),
        CANDIDATES[1]: TimeSeries(
            TIMESTAMPS,
            [float(30 - step + (step % 3)) for step in TIMESTAMPS],
        ),
    }


def _association_config() -> dict[str, object]:
    config = copy.deepcopy(DEFAULT_ASSOCIATION_CONFIG)
    config.update(
        {
            "context_ratio": 1.0,
            "min_aligned_points": 5,
            "min_rf_samples": 100,
            "min_coverage_ratio": 0.5,
        }
    )
    return config


def _candidate_case(
    candidate_count: int,
) -> tuple[dict[str, object], dict[str, TimeSeries], list[str]]:
    candidates = [f"candidate_{index}" for index in range(candidate_count)]
    metrics = [TARGET, *candidates]
    detection: dict[str, object] = {
        "taskId": "top-k-test",
        "states": {metric: 0 for metric in metrics},
        "results": {
            TARGET: {"pointDiagnostics": _diagnostics({8, 9, 10, 11})},
            **{
                metric: {"pointDiagnostics": _diagnostics({9, 10})}
                for metric in candidates
            },
        },
        "abnormalTimeRange": {
            TARGET: [{"startTime": 8.0, "endTime": 11.0}],
            **{metric: [] for metric in candidates},
        },
    }
    inference = {
        TARGET: TimeSeries(TIMESTAMPS, [float(step) for step in TIMESTAMPS]),
        **{
            metric: TimeSeries(
                TIMESTAMPS,
                [
                    float((index + 2) * step + (step % (index + 2)))
                    for step in TIMESTAMPS
                ],
            )
            for index, metric in enumerate(candidates)
        },
    }
    return detection, inference, candidates


def _write_inference_log(
    path: Path,
    inference: dict[str, TimeSeries],
) -> None:
    metrics = list(inference)
    path.write_text(
        "".join(
            f"step:{step} - "
            + " - ".join(
                f"{metric}:{inference[metric].values[step]}" for metric in metrics
            )
            + "\n"
            for step in TIMESTAMPS
        ),
        encoding="utf-8",
    )


def test_detect_summary_returns_one_abnormal_metric_and_event():
    summary = build_abnormal_metrics(_detection_result())

    assert summary == [
        {
            "metric": TARGET,
            "events": [
                {"eventIndex": 0, "startTime": 8.0, "endTime": 11.0}
            ],
        }
    ]


def test_detect_summary_preserves_multiple_event_order():
    result = _detection_result(
        [
            {"startTime": 2.0, "endTime": 3.0},
            {"startTime": 8.0, "endTime": 11.0},
        ]
    )

    summary = build_abnormal_metrics(result)

    assert summary[0]["events"] == [
        {"eventIndex": 0, "startTime": 2.0, "endTime": 3.0},
        {"eventIndex": 1, "startTime": 8.0, "endTime": 11.0},
    ]


def test_detect_summary_is_empty_and_terminal_is_clear_without_abnormality():
    result = _detection_result([])
    attach_abnormal_metrics(result)

    assert result["abnormalMetrics"] == []
    assert "No abnormal metrics detected." in format_terminal_summary(result)


def test_terminal_lists_abnormal_metric_and_events():
    result = _detection_result(
        [
            {"startTime": 104.0, "endTime": 109.0},
            {"startTime": 205.0, "endTime": 220.0},
        ]
    )
    attach_abnormal_metrics(result)

    terminal = format_terminal_summary(result)

    assert "Abnormal metrics:" in terminal
    assert TARGET in terminal
    assert "  event 0: 104 -> 109" in terminal
    assert "  event 1: 205 -> 220" in terminal


def test_analyze_selects_requested_metric_and_event_without_mutating_result():
    detection = _detection_result(
        [
            {"startTime": 2.0, "endTime": 3.0},
            {"startTime": 8.0, "endTime": 11.0},
        ]
    )
    before = copy.deepcopy(detection)

    analysis = analyze_detection_event(
        detection,
        _inference(),
        target_metric=TARGET,
        event_index=1,
        top_k=1,
        association_config=_association_config(),
    )

    assert analysis["targetMetric"] == TARGET
    assert analysis["eventIndex"] == 1
    assert analysis["abnormalRange"] == {"startTime": 8.0, "endTime": 11.0}
    assert analysis["analysisWindow"] == {
        "startTime": 5.0,
        "endTime": 14.0,
        "usedMinimumContext": False,
    }
    assert detection == before


def test_analyze_returns_top_k_with_existing_evidence_fields():
    analysis = analyze_detection_event(
        _detection_result(),
        _inference(),
        target_metric=TARGET,
        event_index=0,
        top_k=1,
        association_config=_association_config(),
    )

    assert len(analysis["topAssociations"]) == 1
    association = analysis["topAssociations"][0]
    assert set(association) >= {
        "rank",
        "metric",
        "abnormalContribution",
        "pearson",
        "spearman",
        "randomForestImportance",
        "coverageRatio",
        "alignedSampleCount",
    }
    assert json.loads(serialize_result(analysis)) == analysis


def test_top_k_above_candidate_count_returns_every_candidate():
    analysis = analyze_detection_event(
        _detection_result(),
        _inference(),
        target_metric=TARGET,
        event_index=0,
        top_k=5,
        association_config=_association_config(),
    )

    assert len(analysis["topAssociations"]) == len(CANDIDATES)
    assert [item["rank"] for item in analysis["topAssociations"]] == [1, 2]
    assert {
        item["metric"] for item in analysis["topAssociations"]
    } == set(CANDIDATES)


def test_analyze_rejects_missing_metric():
    with pytest.raises(ValueError, match="absent from the detection result"):
        analyze_detection_event(
            _detection_result(),
            _inference(),
            target_metric="missing",
            event_index=0,
            top_k=5,
            association_config=_association_config(),
        )


def test_analyze_rejects_metric_without_abnormal_events():
    with pytest.raises(ValueError, match="has no abnormal events"):
        analyze_detection_event(
            _detection_result(),
            _inference(),
            target_metric=CANDIDATES[0],
            event_index=0,
            top_k=5,
            association_config=_association_config(),
        )


def test_analyze_rejects_event_index_out_of_range():
    with pytest.raises(ValueError, match="event-index 2 is out of range"):
        analyze_detection_event(
            _detection_result(),
            _inference(),
            target_metric=TARGET,
            event_index=2,
            top_k=5,
            association_config=_association_config(),
        )


def test_analyze_cli_writes_json_and_preserves_detection_file(tmp_path: Path):
    result_path = tmp_path / "detection.json"
    inference_path = tmp_path / "inference.log"
    policy_path = tmp_path / "algorithm.yaml"
    output_path = tmp_path / "association.json"
    detection_text = json.dumps(_detection_result())
    result_path.write_text(detection_text, encoding="utf-8")
    inference = _inference()
    inference_path.write_text(
        "".join(
            f"step:{step} - "
            + " - ".join(
                f"{metric}:{inference[metric].values[step]}"
                for metric in [TARGET, *CANDIDATES]
            )
            + "\n"
            for step in TIMESTAMPS
        ),
        encoding="utf-8",
    )
    policy_path.write_text(
        "association:\n"
        "  context_ratio: 1.0\n"
        "  min_aligned_points: 5\n"
        "  min_rf_samples: 100\n"
        "  min_coverage_ratio: 0.5\n",
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "analyze",
            "--result",
            str(result_path),
            "--inference-log",
            str(inference_path),
            "--target-metric",
            TARGET,
            "--event-index",
            "0",
            "--top-k",
            "1",
            "--policy-config",
            str(policy_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert result_path.read_text(encoding="utf-8") == detection_text
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["targetMetric"] == TARGET
    assert output["eventIndex"] == 0
    assert len(output["topAssociations"]) == 1


def test_analyze_cli_top_five_is_exact_prefix_of_six_ranked_candidates(
    tmp_path: Path,
):
    detection, inference, candidates = _candidate_case(6)
    result_path = tmp_path / "detection.json"
    inference_path = tmp_path / "inference.log"
    policy_path = tmp_path / "algorithm.yaml"
    output_path = tmp_path / "association.json"
    result_path.write_text(json.dumps(detection), encoding="utf-8")
    _write_inference_log(inference_path, inference)
    policy_path.write_text(
        "association:\n"
        "  context_ratio: 1.0\n"
        "  min_aligned_points: 5\n"
        "  min_rf_samples: 100\n"
        "  min_coverage_ratio: 0.5\n",
        encoding="utf-8",
    )
    full_ranking = analyze_detection_event(
        detection,
        inference,
        target_metric=TARGET,
        event_index=0,
        top_k=len(candidates),
        association_config=_association_config(),
    )["topAssociations"]

    exit_code = cli.main(
        [
            "analyze",
            "--result",
            str(result_path),
            "--inference-log",
            str(inference_path),
            "--target-metric",
            TARGET,
            "--event-index",
            "0",
            "--top-k",
            "5",
            "--policy-config",
            str(policy_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    output = json.loads(output_path.read_text(encoding="utf-8"))
    top_associations = output["topAssociations"]
    assert len(full_ranking) == 6
    assert len(top_associations) == 5
    assert [item["rank"] for item in top_associations] == [1, 2, 3, 4, 5]
    assert top_associations == full_ranking[:5]
