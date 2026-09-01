# Degradation Algorithm Contract

Use this concise reference to preserve the semantics implemented in
`experiment/degradation`. For implementation details, consult
`docs/algorithm.md` and the source modules named below.

## Contract Summary

```text
Prometheus series
  -> global-step alignment
  -> KDE baseline training
  -> point classification and 3-of-5 event detection
  -> correlation/random-forest association
  -> persisted baseline and event records
```

## 1. Data Input and Step Alignment

### Input

- Read samples from Prometheus.
- Use `rl_insight_monitor_training_global_step` as the step ruler.
- Discover metrics through the configured series selector.

### Rules

- Require the instant global-step query to return exactly one finite,
  integer-valued series.
- Define step `k` as `[timestamp(k), timestamp(k + 1))` and use the last finite
  metric sample inside that interval.
- Preserve a missing value as `None`; do not drop the complete step.
- Require global steps and the final boundary to be consecutive. Do not infer,
  interpolate, or silently skip a missing step.
- Keep series with different label sets separate even when their metric names
  match.
- Skip catalog metrics that the selected training task does not expose.
- Model 15 scalar latency targets with an `UP` policy and 95 scalar candidates
  with a `BOTH` policy.
- Keep cataloged raw Histogram targets and Counter candidates out of the current
  KDE pipeline.
  Aggregate Histograms and use Prometheus `rate()` or `increase()` for Counters
  before a future integration admits them.

### Output

Produce step-indexed `StepFrame` objects without collapsing concrete labeled
series or hiding missing observations.

## 2. Baseline Training

### Input

Collect complete aligned steps for every discovered concrete series.

### Algorithm

- Fit a Gaussian KDE, find density modes, map them back to step-contiguous
  segments, and retain only segments that pass the configured stability checks.
- Fit each stable segment independently and derive normal bounds from the KDE
  CDF tail probability.
- Expand lower and upper thresholds outward according to sign.
- Preserve multiple independent normal ranges for multimodal metrics.

### Default Parameters

| Parameter | Default |
|---|---:|
| Baseline collection | 30 complete steps |
| Minimum finite samples | 20 per concrete series |
| Event evidence window | 5 valid target points |
| Event evidence threshold | 3 of 5 |
| Pre-event context | 30 steps |
| Correlation weight | 0.85 |
| Random-forest weight | 0.15 |
| Association result limit | Top-25 |

### Output

Persist the fitted normal ranges as the frozen baseline. Reuse that baseline
across restarts until the user explicitly resets it. Require at least one fitted
target series and one fitted candidate series before detection begins.

## 3. Anomaly and Event Detection

### Point States

Classify each point relative to all fitted normal ranges:

| State | Meaning |
|---|---|
| `NORMAL` | Inside a normal range, including its threshold boundaries |
| `UP` | Above all applicable upper thresholds |
| `DOWN` | Below all applicable lower thresholds |
| `BETWEEN_MODES` | Outside the retained ranges between independent modes |

- For targets, count only `UP` as abnormal.
- For candidates, count `UP`, `DOWN`, and `BETWEEN_MODES` as abnormal.

### Event Lifecycle

| Evidence window | Transition |
|---|---|
| At least three abnormal points in five consecutive valid target points | Confirm the target event |
| Fewer than three abnormal points in the same complete window after confirmation | Close the active event |
| Any point is `None` | Make no transition until five consecutive valid points are available again |

### Output

Emit deterministic `confirmed` and `closed` lifecycle transitions for target
events. Candidate point states are evidence for association; they do not create
target events.

## 4. Association Analysis

### Trigger and Input

- Analyze once when an event is confirmed and again when the same event closes.
- Align target and candidate values by identical global steps.
- Delete missing pairs only for the affected candidate.
- Admit to random-forest analysis only candidates with point-level abnormality
  in the event interval.

### Scoring

- Calculate Pearson and Spearman correlation and use the coefficient with the
  greater absolute magnitude while preserving its sign.
- Use a chronological 70/30 train/validation split, balanced accuracy, and
  permutation importance for random-forest evidence.
- Return an explicit reason when coverage, common samples, or class diversity is
  insufficient. Do not fall back to impurity importance or a full-data split.
- Normalize correlation strength and non-negative forest importance across the
  valid candidate set.
- Combine the normalized evidence with default effective weights of 0.85 for
  correlation and 0.15 for random forest. When only one evidence path is valid,
  give that path full effective weight.

### Output

Return the Top-25 associations by default. Interpret `association_score` as
relative association evidence within the current candidate set, never as
causality, degradation magnitude, or an independently calibrated probability.

## 5. Persistence and Runtime Scope

### State Files

| File | Stored content |
|---|---|
| `standard_data.json` | Frozen baseline, schema version 1 |
| `abnormal_data.json` | Confirmed and closed event records, schema version 1 |

Retain 30 pre-event context steps. The baseline file stores the fitted result,
not the original 30-step observations.

### Runtime Boundaries

- Run as a source-tree experiment for one active training task.
- Do not imply that it is packaged in the published wheel or wired into the root
  `rl-insight` CLI.
- Treat request retry, multi-task isolation, dynamic re-discovery after startup,
  alert delivery, and service supervision as out of scope.

For implementation details, inspect
`experiment/degradation/{metrics,window,baseline,detector,association,runtime,storage}.py`
and the matching tests under `tests/monitor/ut/degradation`.
