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

import copy
from typing import Any

import pytest

from experiment.degradation_perception.policy import (
    DEFAULT_METRIC_POLICY,
    resolve_history_policy,
    resolve_metric_policy,
)


METRIC = 'timing_s/step'


def test_default_policy_preserves_existing_algorithm_values():
    policy = resolve_metric_policy(METRIC)
    assert policy['metric'] == METRIC
    assert policy['alpha'] == 0.01
    assert policy['upper_ratio'] == 1.15
    assert policy['lower_ratio'] == 1.15
    assert policy['minimum_standard_points'] == 3
    assert policy['minimum_inference_points'] == 5
    assert policy['kde']['random_seed'] == 42
    assert policy['stable_segment']['minimum_passed_flags'] == 4
    assert policy['abnormal_interval']['minimum_abnormal_points'] == 5
    assert policy['association']['enabled'] is False


def test_partial_policy_override_is_deep_merged_without_mutating_defaults():
    policy = resolve_metric_policy(
        METRIC,
        {
            'association': {
                'enabled': True,
                'target_metrics': [METRIC],
                'weights': {'correlation': 0.7, 'random_forest': 0.3},
            }
        },
    )
    assert policy['association']['enabled'] is True
    assert policy['association']['weights'] == {
        'correlation': 0.7,
        'random_forest': 0.3,
    }
    assert policy['association']['random_forest']['n_estimators'] == 200
    assert DEFAULT_METRIC_POLICY['association']['enabled'] is False


def _set_path(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


@pytest.mark.parametrize(
    ('path', 'value', 'match'),
    [
        (('association', 'enabled'), 1, 'enabled'),
        (('association', 'target_metrics'), METRIC, 'target_metrics'),
        (('association', 'target_metrics'), [''], 'target_metrics'),
        (('association', 'candidate_mode'), 'all', 'candidate_mode'),
        (('association', 'weights', 'correlation'), -0.1, 'correlation'),
        (('association', 'weights', 'random_forest'), 0.6, 'sum to 1'),
        (('association', 'top_k'), 0, 'top_k'),
        (('association', 'context_ratio'), -1.0, 'context_ratio'),
        (('association', 'min_aligned_points'), 0, 'min_aligned_points'),
        (('association', 'min_rf_samples'), 0, 'min_rf_samples'),
        (('association', 'min_coverage_ratio'), 1.1, 'min_coverage_ratio'),
        (('association', 'alignment_tolerance'), 0.0, 'alignment_tolerance'),
        (('association', 'random_forest', 'n_estimators'), 0, 'n_estimators'),
        (('association', 'random_forest', 'class_weight'), None, 'class_weight'),
        (('association', 'random_forest', 'random_state'), -1, 'random_state'),
        (
            ('association', 'random_forest', 'importance_method'),
            'impurity',
            'importance_method',
        ),
    ],
)
def test_invalid_association_values_are_rejected(path, value, match):
    policy = copy.deepcopy(DEFAULT_METRIC_POLICY)
    policy['metric'] = METRIC
    _set_path(policy, path, value)
    with pytest.raises(ValueError, match=match):
        resolve_metric_policy(METRIC, policy)


@pytest.mark.parametrize(
    ('section', 'key'),
    [
        (('association',), 'unknown'),
        (('association', 'weights'), 'unknown'),
        (('association', 'random_forest'), 'unknown'),
    ],
)
def test_unknown_association_keys_are_rejected(section, key):
    policy = copy.deepcopy(DEFAULT_METRIC_POLICY)
    policy['metric'] = METRIC
    target = policy
    for part in section:
        target = target[part]
    target[key] = 1
    with pytest.raises(ValueError, match='unknown keys'):
        resolve_metric_policy(METRIC, policy)


@pytest.mark.parametrize(
    ('override', 'match'),
    [
        ({'alpha': 0.5}, 'alpha'),
        ({'upper_ratio': 0.9}, 'upper_ratio'),
        ({'minimum_standard_points': 2}, 'minimum_standard_points'),
        ({'normalization': {'type': 'zscore'}}, 'normalization'),
        ({'kde': {'kernel': 'epanechnikov'}}, 'kernel'),
        ({'stable_segment': {'minimum_passed_flags': 7}}, 'minimum_passed_flags'),
        ({'abnormal_interval': {'minimum_abnormal_rate': 1.1}}, 'abnormal_rate'),
    ],
)
def test_invalid_detection_policy_values_are_rejected(override, match):
    with pytest.raises(ValueError, match=match):
        resolve_metric_policy(METRIC, override)


def test_history_policy_defaults_and_validation():
    assert resolve_history_policy() == {
        'n_keep_result': 1,
        'n_keep_abnormal': 1,
    }
    assert resolve_history_policy(
        {'n_keep_result': 3, 'n_keep_abnormal': 2}
    ) == {'n_keep_result': 3, 'n_keep_abnormal': 2}
    with pytest.raises(ValueError, match='must not exceed'):
        resolve_history_policy({'n_keep_result': 1, 'n_keep_abnormal': 2})
    with pytest.raises(ValueError, match='unknown keys'):
        resolve_history_policy({'unknown': 1})
