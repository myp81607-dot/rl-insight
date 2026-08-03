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
from experiment.degradation_perception.training_log import parse_verl_training_log


def test_step_zero_and_all_required_numeric_formats_are_supported():
    parsed = parse_verl_training_log(
        "step:0 - integer:7 - float.value:1.25 - negative/value:-3.5 "
        "- scientific_value:6.02e-3"
    )

    assert parsed == {
        "integer": TimeSeries([0.0], [7.0]),
        "float.value": TimeSeries([0.0], [1.25]),
        "negative/value": TimeSeries([0.0], [-3.5]),
        "scientific_value": TimeSeries([0.0], [0.00602]),
    }
