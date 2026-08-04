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

from pathlib import Path

import pytest

from experiment.degradation_perception.algorithm_policy import (
    DEFAULT_ALGORITHM_POLICY_PATH,
    load_algorithm_policy_config,
)


METRIC = "timing_s/step"


def test_default_yaml_exposes_requested_algorithm_parameters():
    config = load_algorithm_policy_config(DEFAULT_ALGORITHM_POLICY_PATH)
    policy = config.for_metrics([METRIC])[METRIC]

    assert policy["abnormal_type"] == "UP"
    assert policy["alpha"] == pytest.approx(0.01)
    assert policy["upper_ratio"] == pytest.approx(1.15)
    assert policy["lower_ratio"] == pytest.approx(1.15)
    assert policy["kde"] == {
        "bandwidth": "auto",
        "peak_prominence_ratio": 0.01,
    }
    assert policy["stable_segment"] == {
        "std_factor": 2.0,
        "minimum_passed_flags": 4,
    }
    assert policy["abnormal_interval"] == {
        "minimum_duration": 0.5,
        "minimum_abnormal_points": 5,
        "minimum_abnormal_rate": 0.60,
        "max_normal_points_between": 1,
    }
    assert config.history_policy == {
        "n_keep_result": 1,
        "n_keep_abnormal": 1,
    }
    assert policy["association"]["weights"] == {
        "correlation": 0.5,
        "random_forest": 0.5,
    }
    assert policy["association"]["top_k"] == 5
    assert policy["association"]["context_ratio"] == pytest.approx(1.0)
    assert policy["association"]["min_aligned_points"] == 10
    assert policy["association"]["min_rf_samples"] == 30
    assert policy["association"]["min_coverage_ratio"] == pytest.approx(0.6)


def test_custom_yaml_is_validated_and_reused_for_each_metric(tmp_path: Path):
    path = tmp_path / "algorithm.yaml"
    path.write_text(
        "alpha: 0.2\n"
        "history_policy:\n"
        "  n_keep_result: 3\n"
        "  n_keep_abnormal: 2\n"
        "association:\n"
        "  weights:\n"
        "    correlation: 0.25\n"
        "    random_forest: 0.75\n",
        encoding="utf-8",
    )

    config = load_algorithm_policy_config(path)

    assert config.for_metrics([METRIC, "actor/entropy"]) == {
        METRIC: {
            "alpha": 0.2,
            "association": {
                "weights": {"correlation": 0.25, "random_forest": 0.75}
            },
        },
        "actor/entropy": {
            "alpha": 0.2,
            "association": {
                "weights": {"correlation": 0.25, "random_forest": 0.75}
            },
        },
    }
    assert config.history_policy == {
        "n_keep_result": 3,
        "n_keep_abnormal": 2,
    }


def test_unknown_yaml_parameter_is_rejected(tmp_path: Path):
    path = tmp_path / "algorithm.yaml"
    path.write_text("alphaa: 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown keys"):
        load_algorithm_policy_config(path)
