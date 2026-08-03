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

from experiment.degradation_perception.perception_config import TimeSeries
from experiment.degradation_perception.training_log import (
    parse_verl_training_log,
    parse_verl_training_log_line,
    parse_verl_training_log_with_metadata,
)


def test_requires_step_key_in_the_first_field():
    parsed = parse_verl_training_log(
        "prefix step:1 - timing_s/step:1.0\n"
        "global_step:2 - timing_s/step:2.0\n"
        "step:3 - timing_s/step:3.0"
    )

    assert parsed == {"timing_s/step": TimeSeries([3.0], [3.0])}


def test_line_parser_uses_exact_step_and_first_colon_splits():
    assert parse_verl_training_log_line(
        "step:7 - actor/entropy:4.25 - bad:1:2"
    ) == (7, {"actor/entropy": 4.25})
    assert parse_verl_training_log_line("step:not-an-int - metric:1") is None
    assert parse_verl_training_log_line("step:8 - metric:nan") is None


def test_metadata_api_filters_only_steps_above_last_step():
    parsed = parse_verl_training_log_with_metadata(
        "step:1 - metric:1.0\n"
        "step:3 - metric:3.0\n"
        "step:4 - invalid:not-numeric\n"
        "step:5 - other:5.0",
        min_step_exclusive=2,
    )

    assert parsed.valid_steps == [3, 5]
    assert parsed.series == {
        "metric": TimeSeries([3.0], [3.0]),
        "other": TimeSeries([5.0], [5.0]),
    }


def test_duplicate_step_still_uses_shared_last_value_rule():
    parsed = parse_verl_training_log_with_metadata(
        "step:2 - metric:2.0\nstep:2 - metric:20.0"
    )

    assert parsed.valid_steps == [2, 2]
    assert parsed.series["metric"] == TimeSeries([2.0], [20.0])
