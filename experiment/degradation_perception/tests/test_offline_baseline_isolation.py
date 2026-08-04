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

from experiment.degradation_perception.detection_runtime import DetectionRunner
from experiment.degradation_perception.perception_config import TimeSeries


def test_inference_values_never_change_the_healthy_kde_thresholds():
    metric = "timing_s/step"
    healthy = [1.00, 1.01, 0.99, 1.02, 1.00, 0.98] * 5
    standard = {
        metric: TimeSeries(list(range(len(healthy))), healthy),
    }
    normal_inference = {
        metric: TimeSeries(list(range(6)), [1.0, 1.01, 0.99, 1.0, 1.02, 0.98])
    }
    faulty_inference = {
        metric: TimeSeries(list(range(6)), [50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    }

    normal_result = DetectionRunner(
        standard=standard,
        metrics=[metric],
    ).run(normal_inference)
    faulty_result = DetectionRunner(
        standard=standard,
        metrics=[metric],
    ).run(faulty_inference)

    normal_models = normal_result["debugDetails"]["metrics"][metric]["normalModels"]
    faulty_models = faulty_result["debugDetails"]["metrics"][metric]["normalModels"]
    assert normal_models == faulty_models
