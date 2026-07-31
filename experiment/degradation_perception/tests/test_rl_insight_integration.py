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

import argparse
import json

import pytest

from experiment.degradation_perception import cli as cli_module
from experiment.degradation_perception.cli import add_degradation_parser
from experiment.degradation_perception.rl_insight_integration import (
    DerivedMetricPublisher,
    MetricQuerySpec,
    PrometheusDegradationRunner,
    derive_detection_windows,
    resolve_prometheus_endpoint,
)


def _matrix(metric: str, start: float, values: list[float]) -> dict:
    return {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {"__name__": metric},
                    "values": [
                        [start + index * 10, str(value)]
                        for index, value in enumerate(values)
                    ],
                }
            ],
        },
    }


def test_prometheus_endpoint_uses_public_service_discovery():
    endpoint = resolve_prometheus_endpoint(
        rl_insight_server_url="http://monitor.example:18080",
        service_fetcher=lambda: {
            "status": "ok",
            "prometheus_port": 9090,
        },
    )
    assert endpoint == "http://monitor.example:9090"


def test_prometheus_endpoint_prefers_public_explicit_endpoint_field():
    endpoint = resolve_prometheus_endpoint(
        rl_insight_server_url="http://monitor.example:18080",
        service_fetcher=lambda: {
            "status": "ok",
            "prometheus_port": 9090,
            "prometheus_endpoint": "https://prom.example/internal",
        },
    )
    assert endpoint == "https://prom.example/internal"


def test_development_override_does_not_call_discovery():
    endpoint = resolve_prometheus_endpoint(
        override="http://127.0.0.1:19090/",
        service_fetcher=lambda: pytest.fail("discovery must not run"),
    )
    assert endpoint == "http://127.0.0.1:19090"


def test_window_priority_explicit_then_metadata_then_defaults():
    explicit = derive_detection_windows(
        now=1000,
        standard_start=100,
        standard_end=200,
        inference_start=300,
        inference_end=400,
    )
    assert explicit.source == "explicit_cli_or_api"
    assert explicit.standard.to_dict() == {"start": 100.0, "end": 200.0}
    assert explicit.inference.to_dict() == {"start": 300.0, "end": 400.0}

    metadata = derive_detection_windows(
        now=1000,
        task_metadata={"start_time": 800},
        baseline_window_seconds=300,
        inference_window_seconds=100,
    )
    assert metadata.source == "task_metadata"
    assert metadata.standard.start == 800
    assert metadata.standard.end == 900
    assert metadata.inference.start == 900
    assert metadata.inference.end == 1000

    defaults = derive_detection_windows(
        now=1000,
        baseline_window_seconds=300,
        inference_window_seconds=100,
    )
    assert defaults.source == "degradation_defaults"
    assert defaults.standard.to_dict() == {"start": 600.0, "end": 900.0}
    assert defaults.inference.to_dict() == {"start": 900.0, "end": 1000.0}


def test_query_spec_uses_unified_labels_and_advanced_template():
    labels = {
        "project": "verl",
        "experiment_name": 'run"1',
        "worker": "trainer_0",
    }
    direct = MetricQuerySpec("step", "rl_insight_monitor_timing_s_step")
    query = direct.query_for(labels)
    assert query.startswith("rl_insight_monitor_timing_s_step{")
    assert 'experiment_name="run\\"1"' in query

    rate = MetricQuerySpec(
        "latency",
        "request_latency_seconds",
        "rate({{metric}}_sum{{labels}}[1m]) "
        "/ rate({{metric}}_count{{labels}}[1m])",
    )
    rendered = rate.query_for(labels)
    assert "request_latency_seconds_sum{" in rendered
    assert rendered.count('project="verl"') == 2


def test_runner_queries_both_windows_and_returns_full_detection_response(tmp_path):
    calls = []

    class FakeClient:
        def __init__(self, endpoint, **kwargs):
            assert endpoint == "http://prometheus.test:9090"

        def query_range(self, query, *, start, end, step):
            calls.append((query, start, end, step))
            values = [1.0, 1.01, 0.99, 1.0, 1.02, 0.98]
            return _matrix("metric_a", start, values)

    class FakeDetector:
        def __init__(self, **kwargs):
            self.dataset = kwargs["dataset"]

        def detect(self):
            assert set(self.dataset) == {"standard", "inference"}
            return {
                "taskId": "run-a",
                "states": {"logical": 0},
                "results": {"logical": {"thresholds": []}},
                "abnormalTimeRange": {"logical": []},
            }

    runner = PrometheusDegradationRunner(
        metric_specs=[MetricQuerySpec("logical", "metric_a")],
        project="verl",
        experiment_name="run-a",
        endpoint_override="http://prometheus.test:9090",
        query_client_factory=FakeClient,
        detector_factory=FakeDetector,
        config_dir=tmp_path,
    )
    response = runner.run_once(
        standard_start=100,
        standard_end=200,
        inference_start=300,
        inference_end=400,
    )
    assert response["abnormalTimeRange"] == {"logical": []}
    assert len(calls) == 2
    assert 'project="verl"' in calls[0][0]
    assert 'experiment_name="run-a"' in calls[1][0]
    assert runner.last_diagnostics["windows"]["source"] == "explicit_cli_or_api"


class _FakeRegistry:
    def __init__(self):
        self.values = []

    def value(self, name, documentation, value, defaults, labels):
        self.values.append((name, value, dict(defaults), dict(labels)))


def test_derived_metrics_are_bounded_and_do_not_label_intervals():
    registry = _FakeRegistry()
    publisher = DerivedMetricPublisher(
        {
            "project": "verl",
            "experiment_name": "run-a",
            "worker": "trainer_0",
            "replica": "",
        },
        registry=registry,
    )
    response = {
        "states": {"timing_s/step": 0},
        "results": {
            "timing_s/step": {
                "thresholds": [
                    {"lower_threshold": 0.8, "upper_threshold": 1.2}
                ],
                "currentAbnormalTimeRange": [
                    {
                        "startTime": 100,
                        "endTime": 200,
                        "abnormalRate": 0.75,
                    }
                ],
                "pointDiagnostics": [
                    {"abnormal": True},
                    {"abnormal": False},
                ],
            }
        },
        "abnormalTimeRange": {
            "timing_s/step": [{"startTime": 100, "endTime": 200}]
        },
        "associationAnalysis": {
            "targets": {
                "timing_s/step": {
                    "events": [
                        {
                            "topAssociations": [
                                {
                                    "metric": "latency",
                                    "rank": 1,
                                    "abnormalContribution": 71.5,
                                }
                            ]
                        }
                    ]
                }
            }
        },
    }
    publisher.publish(response, detected_at=500)

    names = {item[0] for item in registry.values}
    assert {
        "abnormal",
        "detection_state",
        "lower_threshold",
        "upper_threshold",
        "abnormal_rate",
        "abnormal_point_count",
        "last_detected_timestamp_seconds",
        "interval_active",
        "association_score_percent",
    } <= names
    all_labels = [
        {**defaults, **labels}
        for _, _, defaults, labels in registry.values
    ]
    assert not any("startTime" in labels for labels in all_labels)
    assert not any("endTime" in labels for labels in all_labels)
    assert not any("query" in labels for labels in all_labels)


def test_metrics_endpoint_reuses_main_registration_helpers():
    registry = _FakeRegistry()
    publisher = DerivedMetricPublisher({}, registry=registry)
    calls = []

    endpoint = publisher.start_endpoint(
        port=9093,
        bind_address="127.0.0.1",
        advertise_host="10.0.0.8",
        server_starter=lambda port, addr: calls.append(("start", port, addr)),
        target_registrar=lambda targets, job_name: calls.append(
            ("register", targets, job_name)
        ),
    )

    assert endpoint == "http://10.0.0.8:9093/metrics"
    assert calls == [
        ("start", 9093, "127.0.0.1"),
        (
            "register",
            ["10.0.0.8:9093"],
            "rl_insight_degradation",
        ),
    ]


def test_cli_registration_matches_root_command_shape():
    parser = argparse.ArgumentParser(prog="rl-insight")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_degradation_parser(subparsers)
    args = parser.parse_args(
        [
            "degradation",
            "analyze",
            "--project",
            "verl",
            "--experiment-name",
            "run-a",
            "--metric",
            "metric_a",
        ]
    )
    assert args.command == "degradation"
    assert args.degradation_command == "analyze"


def test_standalone_analyze_prints_full_json(monkeypatch, capsys):
    response = {
        "taskId": "run-a",
        "states": {"metric_a": 0},
        "results": {},
        "abnormalTimeRange": {"metric_a": []},
    }

    class FakeRunner:
        last_diagnostics = {}

        def run_once(self, **kwargs):
            return response

    monkeypatch.setattr(
        cli_module,
        "_runner_from_args",
        lambda args: FakeRunner(),
    )
    code = cli_module.main(
        [
            "analyze",
            "--project",
            "verl",
            "--experiment-name",
            "run-a",
            "--metric",
            "metric_a",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out) == response
