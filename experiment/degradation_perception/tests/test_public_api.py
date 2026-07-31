# Copyright (c) 2026 verl-project authors.
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json

from experiment.degradation_perception import (
    DegradationPerception,
    DetectionInput,
    DetectionResult,
    TimeSeries,
)


def test_rl_insight_can_call_the_in_memory_algorithm_api():
    metric = 'timing_s/step'
    healthy = [1.00, 1.01, 0.99, 1.02, 1.00, 0.98] * 5
    current = [1.00, 1.01, 0.99, 1.50, 1.51, 1.49, 1.50, 1.52, 1.50]
    data: DetectionInput = {
        'standard': {
            metric: TimeSeries(
                timestamps=list(range(len(healthy))),
                values=healthy,
            )
        },
        'inference': {
            metric: TimeSeries(
                timestamps=list(range(100, 100 + len(current))),
                values=current,
            )
        },
    }

    result: DetectionResult = DegradationPerception(
        dataset=data,
        metrics=[metric],
        source_type='prometheus',
    ).detect()

    assert result['states'][metric] == 0
    assert result['abnormalTimeRange'][metric]
    json.dumps(result, allow_nan=False)
