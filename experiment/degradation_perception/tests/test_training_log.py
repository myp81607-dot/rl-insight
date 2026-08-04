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

import pytest

from experiment.degradation_perception.perception_config import TimeSeries
from experiment.degradation_perception.training_log import (
    load_verl_training_log,
    parse_verl_training_log,
)


def test_parses_step_lines_multiple_metrics_and_preserves_first_seen_order():
    parsed = parse_verl_training_log(
        """
        startup output that is not a metric record
        step:2 - timing_s/step:1.20 - actor/entropy:0.50
        step:1 - actor/entropy:0.40 - critic/rewards/mean:3.0
        """
    )

    assert list(parsed) == [
        "timing_s/step",
        "actor/entropy",
        "critic/rewards/mean",
    ]
    assert parsed["timing_s/step"] == TimeSeries([2.0], [1.2])
    assert parsed["actor/entropy"] == TimeSeries([1.0, 2.0], [0.4, 0.5])
    assert parsed["critic/rewards/mean"] == TimeSeries([1.0], [3.0])


def test_splits_each_field_only_at_first_colon_and_keeps_other_fields():
    parsed = parse_verl_training_log(
        "step:7 - malformed/value:1.2:unexpected - valid/value:2.5"
    )

    assert parsed == {"valid/value": TimeSeries([7.0], [2.5])}


def test_skips_malformed_non_numeric_and_non_finite_metric_values():
    parsed = parse_verl_training_log(
        "step:4 - no_separator - empty: - text:not-a-number - nan:nan "
        "- positive_inf:inf - negative_inf:-inf - valid:1e-3"
    )

    assert parsed == {"valid": TimeSeries([4.0], [0.001])}


def test_sorts_out_of_order_steps_and_last_valid_duplicate_step_wins():
    parsed = parse_verl_training_log(
        """
        step:3 - actor/loss:3.0
        step:1 - actor/loss:1.0
        step:2 - actor/loss:2.0
        step:2 - actor/loss:20.0
        step:2 - actor/loss:not-numeric
        """
    )

    assert parsed["actor/loss"] == TimeSeries(
        timestamps=[1.0, 2.0, 3.0],
        values=[1.0, 20.0, 3.0],
    )


def test_metrics_may_have_different_point_counts_and_names_are_not_normalized():
    parsed = parse_verl_training_log(
        """
        step:1 - Actor-Loss.Raw/Mean:1.0 - timing_s/step:10.0
        step:2 - timing_s/step:11.0
        step:3 - Actor-Loss.Raw/Mean:3.0
        """
    )

    assert list(parsed) == ["Actor-Loss.Raw/Mean", "timing_s/step"]
    assert parsed["Actor-Loss.Raw/Mean"] == TimeSeries([1.0, 3.0], [1.0, 3.0])
    assert parsed["timing_s/step"] == TimeSeries([1.0, 2.0], [10.0, 11.0])


def test_loads_real_log_lines_from_file(tmp_path):
    path = tmp_path / "worker.log"
    path.write_text(
        'step:10 - timing_s/step:1.5\n'
        'step:11 - timing_s/step:1.6\n'
        "(TaskRunner pid=9) step:10 - timing_s/step:1.5\n"
        "(TaskRunner pid=9) step:11 - timing_s/step:1.6\n",
        encoding="utf-8",
    )

    assert load_verl_training_log(path) == {
        "timing_s/step": TimeSeries([10.0, 11.0], [1.5, 1.6])
    }


def test_rejects_non_string_lines():
    with pytest.raises(TypeError, match="training log lines must be strings"):
        parse_verl_training_log(["step:1 - metric:1.0", 2])  # type: ignore[list-item]
