# Degradation algorithm contract

Use this reference to preserve the v0.1 semantics implemented in
`experiment/degradation`.

## Data boundary

- Read samples from Prometheus and use
  `rl_insight_monitor_training_global_step` as the step ruler.
- Require the instant global-step query to return exactly one finite,
  integer-valued series.
- Keep series with different label sets separate even when their metric names
  match.
- Discover metrics through the configured series selector. Skip catalog metrics
  that the selected training task does not expose.
- Model 15 scalar latency targets with an `UP` policy and 95 scalar candidates
  with a `BOTH` policy.
- Keep the cataloged raw Histogram targets and Counter candidates out of v0.1
  KDE. Aggregate Histograms and use Prometheus `rate()` or `increase()` for
  Counters before a future integration admits them.

## Step alignment

- Define step `k` as `[timestamp(k), timestamp(k + 1))`.
- Use the last finite metric sample inside each step.
- Preserve a missing value as `None`; do not drop the complete step.
- Require global steps and the final boundary to be consecutive. Do not infer,
  interpolate, or silently skip a missing step.

## Baseline

- Collect 30 complete steps by default and require at least 20 finite values per
  concrete series.
- Fit a Gaussian KDE, find density modes, map them back to step-contiguous
  segments, and retain only segments that pass the configured stability checks.
- Fit each stable segment independently and derive normal bounds from the KDE
  CDF tail probability.
- Expand lower and upper thresholds outward according to sign. Preserve multiple
  independent normal ranges for multimodal metrics.
- Freeze the fitted baseline across restarts until the user explicitly resets it.

## Point and event detection

- Classify a point as `NORMAL`, `UP`, `DOWN`, or `BETWEEN_MODES` relative to all
  normal ranges. Threshold boundaries are normal.
- Count only `UP` as abnormal for targets. Count `UP`, `DOWN`, and
  `BETWEEN_MODES` as abnormal for candidates.
- Confirm a target event when at least three of five consecutive, valid target
  points are abnormal.
- Close an active event when the same complete five-point window has fewer than
  three abnormal points.
- Treat any `None` in the evidence window as unknown: neither confirm nor close
  until five consecutive valid points are available again.

## Association analysis

- Analyze once on confirmation and again on closure.
- Align target and candidate values by identical global steps and delete missing
  pairs only for the affected candidate.
- Calculate Pearson and Spearman correlation and use the coefficient with greater
  absolute magnitude; preserve its sign.
- Admit to random-forest analysis only candidates with point-level abnormality in
  the event interval. Use chronological 70/30 train/validation splitting,
  balanced accuracy, and permutation importance.
- Return an explicit reason when coverage, common samples, or class diversity is
  insufficient. Do not fall back to impurity importance or a full-data split.
- Normalize correlation strength and non-negative forest importance across the
  valid candidate set. Combine them with default effective weights of 0.85/0.15;
  let the single valid evidence path receive full weight when the other is
  unavailable.
- Interpret the score as relative association evidence, never causality,
  degradation magnitude, or an independently calibrated probability.

## Persistence and runtime scope

- Store a frozen baseline in `standard_data.json` and confirmed/closed events in
  `abnormal_data.json`, both with schema version 1.
- Retain 30 pre-context steps and return Top-25 associations by default.
- Run as a source-tree experiment for one active training task. Do not imply that
  it is packaged in the published wheel or wired into the root `rl-insight` CLI.
- Treat request retry, multi-task isolation, dynamic re-discovery after startup,
  alert delivery, and service supervision as out of scope.

For implementation details, inspect
`experiment/degradation/{metrics,window,baseline,detector,association,runtime,storage}.py`
and the matching tests under `tests/monitor/ut/degradation`.
