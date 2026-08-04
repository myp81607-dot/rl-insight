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

"""Load user-tunable degradation policy from one YAML file."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .policy import resolve_history_policy, resolve_metric_policy


DEFAULT_ALGORITHM_POLICY_PATH = Path(__file__).with_name("algorithm_config.yaml")


@dataclass(frozen=True)
class AlgorithmPolicyConfig:
    """Validated global metric override and history settings."""

    metric_policy: dict[str, Any]
    history_policy: dict[str, int]

    def for_metrics(
        self,
        metrics: Sequence[str] | str,
    ) -> dict[str, dict[str, Any]]:
        selected = [metrics] if isinstance(metrics, str) else list(metrics)
        return {
            str(metric): copy.deepcopy(self.metric_policy)
            for metric in selected
        }


def load_algorithm_policy_config(
    path: str | Path = DEFAULT_ALGORITHM_POLICY_PATH,
) -> AlgorithmPolicyConfig:
    """Load and strictly validate algorithm and history settings from YAML."""

    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"could not load algorithm policy config: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("algorithm policy config root must be an object")

    supplied = dict(raw)
    history_raw = supplied.pop("history_policy", None)
    if history_raw is not None and not isinstance(history_raw, Mapping):
        raise ValueError("history_policy must be an object")
    if "metric" in supplied:
        raise ValueError(
            "algorithm policy config is shared by selected metrics; "
            "remove the metric field"
        )

    # Reuse the algorithm's complete deep validation while retaining only the
    # user-supplied values as overrides for each selected metric.
    resolve_metric_policy("__algorithm_config__", supplied)
    return AlgorithmPolicyConfig(
        metric_policy=copy.deepcopy(supplied),
        history_policy=resolve_history_policy(history_raw),
    )


__all__ = [
    "AlgorithmPolicyConfig",
    "DEFAULT_ALGORITHM_POLICY_PATH",
    "load_algorithm_policy_config",
]
