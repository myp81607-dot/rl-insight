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
from pathlib import Path


def test_dashboard_is_valid_and_covers_required_views():
    path = (
        Path(__file__).parents[1]
        / "grafana"
        / "degradation_dashboard.json"
    )
    dashboard = json.loads(path.read_text(encoding="utf-8"))
    titles = {panel["title"] for panel in dashboard["panels"]}
    assert {
        "Raw Performance and Detection Thresholds",
        "Current Degradation",
        "Detection State",
        "Abnormal Rate",
        "Abnormal Point Count",
        "Abnormal Activity Periods",
        "Last Detection",
        "Association Analysis Top-K",
    } <= titles

    expressions = [
        target["expr"]
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
    ]
    assert any("rl_insight_degradation_upper_threshold" in expr for expr in expressions)
    assert any("rl_insight_degradation_lower_threshold" in expr for expr in expressions)
    assert any("rl_insight_degradation_interval_active" in expr for expr in expressions)
    assert any(
        "rl_insight_degradation_association_score_percent" in expr
        for expr in expressions
    )
    assert not any("kde" in expr.lower() for expr in expressions)
