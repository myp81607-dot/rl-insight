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

"""Metric catalog for the degradation experiment."""

GLOBAL_STEP_METRIC = "rl_insight_monitor_training_global_step"

# Scalar latency metrics that can be modeled directly with an UP policy.
TARGET_METRICS = (
    "rl_insight_monitor_timing_s_step",
    "rl_insight_monitor_timing_s_gen",
    "rl_insight_monitor_timing_s_ref",
    "rl_insight_monitor_timing_s_adv",
    "rl_insight_monitor_timing_s_old_log_prob",
    "rl_insight_monitor_timing_s_update_actor",
    "rl_insight_monitor_timing_s_update_weights",
    "rl_insight_monitor_timing_s_testing",
    "rl_insight_monitor_timing_per_token_ms_gen",
    "rl_insight_monitor_timing_per_token_ms_ref",
    "rl_insight_monitor_timing_per_token_ms_adv",
    "rl_insight_monitor_timing_per_token_ms_update_actor",
    "rl_insight_monitor_perf_time_per_step",
    "tq_storage_request_latency_p50",
    "tq_storage_request_latency_p99",
)

# Histogram families are cataloged but require aggregation before modeling.
HISTOGRAM_TARGET_METRICS = (
    "vllm:e2e_request_latency_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "tq_controller_request_duration_seconds",
)

# Scalar metrics that can be modeled directly with a BOTH policy.
CANDIDATE_METRICS = (
    "rl_insight_monitor_perf_throughput",
    "rl_insight_monitor_perf_mfu_actor",
    "rl_insight_monitor_perf_total_num_tokens",
    "rl_insight_monitor_actor_perf_cpu_memory_used_gb",
    "rl_insight_monitor_actor_perf_max_memory_allocated_gb",
    "rl_insight_monitor_actor_perf_max_memory_reserved_gb",
    "rl_insight_monitor_prompt_length_mean",
    "rl_insight_monitor_prompt_length_max",
    "rl_insight_monitor_prompt_length_min",
    "rl_insight_monitor_prompt_length_clip_ratio",
    "rl_insight_monitor_response_length_mean",
    "rl_insight_monitor_response_length_max",
    "rl_insight_monitor_response_length_min",
    "rl_insight_monitor_response_length_clip_ratio",
    "rl_insight_monitor_response_length_non_aborted_mean",
    "rl_insight_monitor_response_length_non_aborted_max",
    "rl_insight_monitor_response_length_non_aborted_min",
    "rl_insight_monitor_response_length_non_aborted_clip_ratio",
    "rl_insight_monitor_response_aborted_ratio",
    "rl_insight_monitor_global_seqlen_mean",
    "rl_insight_monitor_global_seqlen_max",
    "rl_insight_monitor_global_seqlen_min",
    "rl_insight_monitor_global_seqlen_minmax_diff",
    "rl_insight_monitor_global_seqlen_balanced_max",
    "rl_insight_monitor_global_seqlen_balanced_min",
    "rl_insight_monitor_actor_loss",
    "rl_insight_monitor_actor_pg_loss",
    "rl_insight_monitor_actor_kl_loss",
    "rl_insight_monitor_actor_entropy_loss",
    "rl_insight_monitor_actor_entropy",
    "rl_insight_monitor_actor_ppo_kl",
    "rl_insight_monitor_actor_kl_coef",
    "rl_insight_monitor_actor_pg_clipfrac",
    "rl_insight_monitor_actor_pg_clipfrac_lower",
    "rl_insight_monitor_actor_grad_norm",
    "rl_insight_monitor_actor_lr",
    "rl_insight_monitor_critic_score_mean",
    "rl_insight_monitor_critic_score_max",
    "rl_insight_monitor_critic_score_min",
    "rl_insight_monitor_critic_rewards_mean",
    "rl_insight_monitor_critic_rewards_max",
    "rl_insight_monitor_critic_rewards_min",
    "rl_insight_monitor_critic_returns_mean",
    "rl_insight_monitor_critic_returns_max",
    "rl_insight_monitor_critic_returns_min",
    "rl_insight_monitor_critic_advantages_mean",
    "rl_insight_monitor_critic_advantages_max",
    "rl_insight_monitor_critic_advantages_min",
    "rl_insight_monitor_training_epoch",
    "rl_insight_monitor_training_num_turns_mean",
    "rl_insight_monitor_training_num_turns_max",
    "rl_insight_monitor_training_num_turns_min",
    "rl_insight_monitor_training_off_policy_trajectory_staleness_mean",
    "rl_insight_monitor_training_off_policy_trajectory_staleness_max",
    "rl_insight_monitor_training_off_policy_trajectory_staleness_worst_mean",
    "rl_insight_monitor_training_off_policy_trajectory_staleness_worst_max",
    "rl_insight_monitor_training_off_policy_trajectory_staleness_worst_min",
    "rl_insight_monitor_training_off_policy_trajectory_spans_mean",
    "rl_insight_monitor_training_off_policy_trajectory_spans_max",
    "rl_insight_monitor_training_off_policy_trajectory_spans_min",
    "rl_insight_monitor_training_rollout_probs_diff_mean",
    "rl_insight_monitor_training_rollout_probs_diff_max",
    "rl_insight_monitor_training_rollout_probs_diff_std",
    "rl_insight_monitor_training_rollout_probs_diff_valid",
    "rl_insight_monitor_training_rollout_actor_probs_pearson_corr",
    "rl_insight_monitor_rollout_corr_kl",
    "rl_insight_monitor_rollout_corr_k3_kl",
    "rl_insight_monitor_rollout_corr_chi2_seq",
    "rl_insight_monitor_rollout_corr_chi2_token",
    "rl_insight_monitor_rollout_corr_log_ppl_diff",
    "rl_insight_monitor_rollout_corr_log_ppl_diff_max",
    "rl_insight_monitor_rollout_corr_log_ppl_diff_min",
    "rl_insight_monitor_rollout_corr_log_ppl_abs_diff",
    "rl_insight_monitor_rollout_corr_ppl_ratio",
    "rl_insight_monitor_rollout_corr_rollout_log_ppl",
    "rl_insight_monitor_rollout_corr_rollout_ppl",
    "rl_insight_monitor_rollout_corr_training_log_ppl",
    "rl_insight_monitor_rollout_corr_training_ppl",
    "rl_insight_monitor_val_aux_num_turns_mean",
    "rl_insight_monitor_val_aux_num_turns_max",
    "rl_insight_monitor_val_aux_num_turns_min",
    "vllm:request_max_num_generation_tokens",
    "vllm:request_prompt_tokens",
    "vllm:request_generation_tokens",
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
    "tq_controller_uptime_seconds",
    "tq_controller_memory_rss_bytes",
    "tq_partition_production_progress",
    "tq_partition_consumption_progress",
    "tq_storage_utilization_ratio",
    "tq_storage_memory_rss_bytes",
    "tq_storage_request_ops",
)

# Raw counters are cataloged but require rate or increase before modeling.
RATE_REQUIRED_CANDIDATE_METRICS = (
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:request_success_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_drafts_total",
    "tq_controller_request_total",
    "tq_controller_request_samples_total",
    "tq_partitions_total",
    "tq_partition_samples_total",
    "tq_storage_active_keys_total",
    "tq_storage_capacity_total",
    "tq_global_index_allocated_total",
    "tq_global_index_reusable_total",
)

CATALOG_TARGET_METRICS = TARGET_METRICS + HISTOGRAM_TARGET_METRICS
CATALOG_CANDIDATE_METRICS = CANDIDATE_METRICS + RATE_REQUIRED_CANDIDATE_METRICS
METRIC_CATALOG = (
    GLOBAL_STEP_METRIC,
    *CATALOG_TARGET_METRICS,
    *CATALOG_CANDIDATE_METRICS,
)

__all__ = [
    "CANDIDATE_METRICS",
    "CATALOG_CANDIDATE_METRICS",
    "CATALOG_TARGET_METRICS",
    "GLOBAL_STEP_METRIC",
    "HISTOGRAM_TARGET_METRICS",
    "METRIC_CATALOG",
    "RATE_REQUIRED_CANDIDATE_METRICS",
    "TARGET_METRICS",
]
