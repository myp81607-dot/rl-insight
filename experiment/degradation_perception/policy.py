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

"""Pure in-memory policies for degradation perception.

The caller owns configuration loading and storage. This module owns the
algorithm defaults, deep merging, and strict validation of supplied values.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from .association_analysis import DEFAULT_ASSOCIATION_CONFIG


DEFAULT_METRIC_POLICY: dict[str, Any] = {
    "metric": "",
    "abnormal_type": "UP",
    "alpha": 0.01,
    "upper_ratio": 1.15,
    "lower_ratio": 1.15,
    "minimum_standard_points": 3,
    "minimum_inference_points": 5,
    "normalization": {
        "type": "identity",
    },
    "kde": {
        "kernel": "gaussian",
        "bandwidth": "auto",
        "grid_size": 1024,
        "padding_ratio": 0.10,
        "tail_bandwidths": 6.0,
        "zero_range_epsilon": 1.0e-8,
        "random_seed": 42,
        "peak_prominence_ratio": 0.01,
    },
    "stable_segment": {
        "std_factor": 2.0,
        "within_std_coefficient": 1.05,
        "minimum_passed_flags": 4,
        "mean_tolerance_ratio": 0.02,
        "time_gap_factor": 3.0,
        "maximum_time_gap": None,
    },
    "abnormal_interval": {
        "minimum_duration": 0.5,
        "minimum_abnormal_points": 5,
        "minimum_abnormal_rate": 0.60,
        "max_normal_points_between": 1,
        "time_gap_factor": 3.0,
        "maximum_time_gap": None,
    },
    "association": copy.deepcopy(DEFAULT_ASSOCIATION_CONFIG),
}

DEFAULT_HISTORY_POLICY: dict[str, int] = {
    "n_keep_result": 1,
    "n_keep_abnormal": 1,
}

_METRIC_POLICY_KEYS = {
    "metric",
    "abnormal_type",
    "alpha",
    "upper_ratio",
    "lower_ratio",
    "minimum_standard_points",
    "minimum_inference_points",
    "normalization",
    "kde",
    "stable_segment",
    "abnormal_interval",
    "association",
}
_NORMALIZATION_KEYS = {"type"}
_KDE_KEYS = {
    "kernel",
    "bandwidth",
    "grid_size",
    "padding_ratio",
    "tail_bandwidths",
    "zero_range_epsilon",
    "random_seed",
    "peak_prominence_ratio",
}
_STABLE_SEGMENT_KEYS = {
    "std_factor",
    "within_std_coefficient",
    "minimum_passed_flags",
    "mean_tolerance_ratio",
    "time_gap_factor",
    "maximum_time_gap",
}
_ABNORMAL_INTERVAL_KEYS = {
    "minimum_duration",
    "minimum_abnormal_points",
    "minimum_abnormal_rate",
    "max_normal_points_between",
    "time_gap_factor",
    "maximum_time_gap",
}
_ASSOCIATION_KEYS = {
    "enabled",
    "target_metrics",
    "candidate_mode",
    "weights",
    "top_k",
    "context_ratio",
    "min_aligned_points",
    "min_rf_samples",
    "min_coverage_ratio",
    "alignment_tolerance",
    "random_forest",
}
_ASSOCIATION_WEIGHT_KEYS = {"correlation", "random_forest"}
_ASSOCIATION_RANDOM_FOREST_KEYS = {
    "n_estimators",
    "class_weight",
    "random_state",
    "importance_method",
}
_HISTORY_POLICY_KEYS = {"n_keep_result", "n_keep_abnormal"}


def resolve_metric_policy(
    metric: str,
    override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a validated policy without reading or writing configuration files."""

    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("metric must be a non-empty string")
    supplied = {} if override is None else dict(override)
    resolved = _deep_merge(DEFAULT_METRIC_POLICY, supplied)
    if supplied.get("metric") in (None, ""):
        resolved["metric"] = metric
    validate_metric_policy(resolved, metric)
    return resolved


def resolve_history_policy(
    override: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Return validated process-local history settings."""

    supplied = {} if override is None else dict(override)
    resolved = _deep_merge(DEFAULT_HISTORY_POLICY, supplied)
    _reject_unknown_keys(resolved, _HISTORY_POLICY_KEYS, "history policy")
    n_keep_result = _require_integer(
        resolved,
        "n_keep_result",
        minimum=1,
        context="history policy",
    )
    n_keep_abnormal = _require_integer(
        resolved,
        "n_keep_abnormal",
        minimum=1,
        context="history policy",
    )
    if n_keep_abnormal > n_keep_result:
        raise ValueError(
            "history policy n_keep_abnormal must not exceed n_keep_result"
        )
    return {
        "n_keep_result": n_keep_result,
        "n_keep_abnormal": n_keep_abnormal,
    }


def validate_metric_policy(config: Mapping[str, Any], metric: str) -> None:
    """Validate one fully resolved per-metric algorithm policy."""

    _reject_unknown_keys(config, _METRIC_POLICY_KEYS, "metric policy")
    if config.get("metric") != metric:
        raise ValueError(
            f"metric policy is bound to {config.get('metric')!r}, "
            f"expected {metric!r}"
        )
    abnormal_type = config.get("abnormal_type")
    if abnormal_type not in {"UP", "DOWN", "BOTH"}:
        raise ValueError("metric policy abnormal_type must be UP, DOWN, or BOTH")
    _require_number(
        config,
        "alpha",
        minimum=0.0,
        maximum=0.5,
        minimum_inclusive=False,
        maximum_inclusive=False,
        context="metric policy",
    )
    _require_number(
        config,
        "upper_ratio",
        minimum=1.0,
        context="metric policy",
    )
    _require_number(
        config,
        "lower_ratio",
        minimum=1.0,
        context="metric policy",
    )
    _require_integer(
        config,
        "minimum_standard_points",
        minimum=3,
        context="metric policy",
    )
    _require_integer(
        config,
        "minimum_inference_points",
        minimum=1,
        context="metric policy",
    )

    normalization = _require_mapping(config, "normalization", "metric policy")
    _reject_unknown_keys(normalization, _NORMALIZATION_KEYS, "normalization")
    if normalization.get("type") != "identity":
        raise ValueError("normalization.type must be identity")

    kde = _require_mapping(config, "kde", "metric policy")
    _reject_unknown_keys(kde, _KDE_KEYS, "kde")
    if kde.get("kernel") != "gaussian":
        raise ValueError("kde.kernel must be gaussian")
    bandwidth = kde.get("bandwidth")
    if bandwidth != "auto":
        _require_finite_number(
            bandwidth,
            "kde.bandwidth",
            minimum=0.0,
            minimum_inclusive=False,
        )
    _require_integer(kde, "grid_size", minimum=32, context="kde")
    _require_number(kde, "padding_ratio", minimum=0.0, context="kde")
    _require_number(
        kde,
        "tail_bandwidths",
        minimum=0.0,
        minimum_inclusive=False,
        context="kde",
    )
    _require_number(
        kde,
        "zero_range_epsilon",
        minimum=0.0,
        minimum_inclusive=False,
        context="kde",
    )
    _require_integer(kde, "random_seed", minimum=0, context="kde")
    _require_number(kde, "peak_prominence_ratio", minimum=0.0, context="kde")

    stable = _require_mapping(config, "stable_segment", "metric policy")
    _reject_unknown_keys(stable, _STABLE_SEGMENT_KEYS, "stable_segment")
    _require_number(
        stable,
        "std_factor",
        minimum=0.0,
        minimum_inclusive=False,
        context="stable_segment",
    )
    _require_number(
        stable,
        "within_std_coefficient",
        minimum=1.0,
        maximum=2.0,
        maximum_inclusive=False,
        context="stable_segment",
    )
    _require_integer(
        stable,
        "minimum_passed_flags",
        minimum=1,
        maximum=6,
        context="stable_segment",
    )
    _require_number(
        stable,
        "mean_tolerance_ratio",
        minimum=0.0,
        context="stable_segment",
    )
    _require_number(
        stable,
        "time_gap_factor",
        minimum=0.0,
        minimum_inclusive=False,
        context="stable_segment",
    )
    _require_optional_positive_number(
        stable.get("maximum_time_gap"),
        "stable_segment.maximum_time_gap",
    )

    interval = _require_mapping(config, "abnormal_interval", "metric policy")
    _reject_unknown_keys(interval, _ABNORMAL_INTERVAL_KEYS, "abnormal_interval")
    _require_number(
        interval,
        "minimum_duration",
        minimum=0.0,
        context="abnormal_interval",
    )
    _require_integer(
        interval,
        "minimum_abnormal_points",
        minimum=1,
        context="abnormal_interval",
    )
    _require_number(
        interval,
        "minimum_abnormal_rate",
        minimum=0.0,
        maximum=1.0,
        context="abnormal_interval",
    )
    _require_integer(
        interval,
        "max_normal_points_between",
        minimum=0,
        context="abnormal_interval",
    )
    _require_number(
        interval,
        "time_gap_factor",
        minimum=0.0,
        minimum_inclusive=False,
        context="abnormal_interval",
    )
    _require_optional_positive_number(
        interval.get("maximum_time_gap"),
        "abnormal_interval.maximum_time_gap",
    )

    association = _require_mapping(config, "association", "metric policy")
    _reject_unknown_keys(association, _ASSOCIATION_KEYS, "association")
    if not isinstance(association.get("enabled"), bool):
        raise ValueError("association.enabled must be a boolean")
    target_metrics = association.get("target_metrics")
    if not isinstance(target_metrics, list):
        raise ValueError("association.target_metrics must be a list")
    if any(
        not isinstance(target, str) or not target.strip()
        for target in target_metrics
    ):
        raise ValueError(
            "association.target_metrics must contain non-empty strings"
        )
    if association.get("candidate_mode") != "abnormal_lower_metrics":
        raise ValueError(
            "association.candidate_mode must be abnormal_lower_metrics"
        )

    weights = _require_mapping(association, "weights", "association")
    _reject_unknown_keys(
        weights,
        _ASSOCIATION_WEIGHT_KEYS,
        "association.weights",
    )
    correlation_weight = _require_number(
        weights,
        "correlation",
        minimum=0.0,
        context="association.weights",
    )
    random_forest_weight = _require_number(
        weights,
        "random_forest",
        minimum=0.0,
        context="association.weights",
    )
    if not math.isclose(
        correlation_weight + random_forest_weight,
        1.0,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    ):
        raise ValueError("association weights must sum to 1")

    _require_integer(association, "top_k", minimum=1, context="association")
    _require_number(
        association,
        "context_ratio",
        minimum=0.0,
        context="association",
    )
    _require_integer(
        association,
        "min_aligned_points",
        minimum=1,
        context="association",
    )
    _require_integer(
        association,
        "min_rf_samples",
        minimum=1,
        context="association",
    )
    _require_number(
        association,
        "min_coverage_ratio",
        minimum=0.0,
        maximum=1.0,
        context="association",
    )
    _require_optional_positive_number(
        association.get("alignment_tolerance"),
        "association.alignment_tolerance",
    )

    random_forest = _require_mapping(
        association,
        "random_forest",
        "association",
    )
    _reject_unknown_keys(
        random_forest,
        _ASSOCIATION_RANDOM_FOREST_KEYS,
        "association.random_forest",
    )
    _require_integer(
        random_forest,
        "n_estimators",
        minimum=1,
        context="association.random_forest",
    )
    if random_forest.get("class_weight") != "balanced":
        raise ValueError(
            "association.random_forest.class_weight must be balanced"
        )
    _require_integer(
        random_forest,
        "random_state",
        minimum=0,
        context="association.random_forest",
    )
    if random_forest.get("importance_method") != "permutation":
        raise ValueError(
            "association.random_forest.importance_method must be permutation"
        )


def _deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    context: str,
) -> None:
    unknown = sorted(repr(key) for key in value if key not in allowed)
    if unknown:
        raise ValueError(f"{context} contains unknown keys: {', '.join(unknown)}")


def _require_mapping(
    value: Mapping[str, Any],
    key: str,
    context: str,
) -> dict[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise ValueError(f"{context}.{key} must be an object")
    return dict(nested)


def _require_integer(
    value: Mapping[str, Any],
    key: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    context: str,
) -> int:
    raw = value.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{context}.{key} must be an integer")
    if minimum is not None and raw < minimum:
        raise ValueError(f"{context}.{key} must be >= {minimum}")
    if maximum is not None and raw > maximum:
        raise ValueError(f"{context}.{key} must be <= {maximum}")
    return raw


def _require_number(
    value: Mapping[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
    context: str,
) -> float:
    return _require_finite_number(
        value.get(key),
        f"{context}.{key}",
        minimum=minimum,
        maximum=maximum,
        minimum_inclusive=minimum_inclusive,
        maximum_inclusive=maximum_inclusive,
    )


def _require_finite_number(
    raw: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(raw)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            operator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{name} must be {operator} {minimum}")
    if maximum is not None:
        invalid = result > maximum if maximum_inclusive else result >= maximum
        if invalid:
            operator = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{name} must be {operator} {maximum}")
    return result


def _require_optional_positive_number(raw: Any, name: str) -> None:
    if raw is None:
        return
    _require_finite_number(
        raw,
        name,
        minimum=0.0,
        minimum_inclusive=False,
    )


__all__ = [
    "DEFAULT_HISTORY_POLICY",
    "DEFAULT_METRIC_POLICY",
    "resolve_history_policy",
    "resolve_metric_policy",
    "validate_metric_policy",
]
