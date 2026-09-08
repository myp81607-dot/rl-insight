---
name: degradation-association
description: "Continuously detect RL training latency degradation from Prometheus, associate candidate metrics with abnormal target events, present the complete grouped Top-25, and infer ranked fault domains and causes."
user_invocable: true
---

# Degradation association

Operate the deterministic implementation in `experiment/degradation`; do not
reimplement its KDE, 3-of-5 event lifecycle, association ranking, persistence,
or presentation logic.

Read [algorithm-contract.md](references/algorithm-contract.md), the concise
skill-facing algorithm contract, before operating the detector or changing
parameters. Consult [docs/algorithm.md](../../../docs/algorithm.md) only when
implementation-level detail is needed. Read
[diagnostic-experience.md](references/diagnostic-experience.md) only for a valid
selected abnormal event. Read [troubleshooting.md](references/troubleshooting.md)
after a failed command or `invalid_state`.

## 1. Environment Check

1. Locate the rl-insight checkout containing `experiment/degradation/cli.py` and
   run commands from that repository root. This experiment is source-tree code,
   not part of the published wheel.
2. Unless the user supplies another path, resolve
   `~/.local/state/rl-insight/degradation/default` to an absolute path and use it
   as `--state-dir`. Use a separate absolute directory for another training task
   and reuse the selected directory for every command.
3. Verify the degradation dependencies. If they are missing, report that fact
   and obtain approval before running `pip install -e ".[degradation]"`.
4. Start with `http://127.0.0.1:9090`. If it is unreachable or the default state
   directory cannot be created or accessed, ask whether the Prometheus URL and
   state directory should be changed. Do not guess another URL, path, or task
   label.

Persisted files are:

```text
<absolute-state-dir>/standard_data.json   # fitted baseline
<absolute-state-dir>/abnormal_data.json   # confirmed/closed events and Top-25
```

## 2. Online Data Access

Run the read-only preflight before `run-once` or `monitor`:

```bash
python -m experiment.degradation.cli check \
  --prometheus-url http://127.0.0.1:9090 \
  --state-dir "$HOME/.local/state/rl-insight/degradation/default"
```

Proceed only when `check` confirms one integer-valued global-step series, at
least one configured target, at least one configured candidate, the required
dependencies, and valid persisted state. Stop on
`error_kind=no_candidate_metrics` and inspect the selector, metric export, and
Prometheus data.

Omit `--series-selector` by default. The core selector `{__name__=~".+"}`
discovers all Prometheus series and then keeps every discovered configured target
and candidate. Never narrow discovery to trainer, job, or another guessed label
set. Use a narrower selector only when the user explicitly supplies or requests
one.

Keep the default event-target set fixed to the eight `timing_s_*` metrics listed
in [algorithm-contract.md](references/algorithm-contract.md). Per-token timing,
`perf_time_per_step`, and transfer-queue request latency are candidate evidence,
not default targets. Do not pass `--target-metric` unless the user explicitly
requests an additional event target.

Use `run-once` only when the user explicitly requests one initialization and
detection poll:

```bash
python -m experiment.degradation.cli run-once \
  --prometheus-url http://127.0.0.1:9090 \
  --state-dir "$HOME/.local/state/rl-insight/degradation/default"
```

Local `show` and `reset` read persisted state and do not require Prometheus.

## 3. RL Latency Anomaly and Association Detection

On a bare invocation of this skill, or when the user asks to start monitoring,
complete Sections 1 and 2, then run `monitor` after a successful `check`. Start
it without an intervening `run-once`:

```bash
python -m experiment.degradation.cli monitor \
  --prometheus-url http://127.0.0.1:9090 \
  --state-dir "$HOME/.local/state/rl-insight/degradation/default"
```

With no baseline, the same process collects 30 complete new steps, writes the
baseline after the next-step boundary, and immediately continues detection. With
an existing baseline, it loads the file and starts detection immediately.
Correlation and random-forest association run at confirmation, refresh once per
monitor poll that yields new complete steps while the same event remains active,
and run once more at closure.

### Continuous monitoring invariant

- Run `monitor` as a foreground process inside one persistent terminal/session
  dedicated to monitoring, and use the conversation as the reporting surface.
  Treat that session as the background monitoring worker; keep `monitor`
  attached rather than daemonizing it.
  If the shell tool exposes a timeout, set it to at least 18,000 seconds
  (5 hours), or unlimited. Treat more than 5 hours as the default monitoring
  scale, not as an automatic stop time.
- Keep waiting on and polling that same session until the user explicitly asks
  to stop. Do not detach, daemonize, replace, or prematurely restart it.
- Do not return a final answer after startup or baseline completion.
- When the log reports `Trained <count> baselines from steps <start>-<end>;
  skipped <count> series`, immediately notify the user with:

  ```text
  Baseline ready: trained <count> baselines from steps <start>-<end>; skipped <count> series. Monitoring continues in the same process.
  ```

- That notification is not completion. Continue watching the same process.
- On every `Confirmed event`, `Updated event`, and `Closed event`, inspect and
  report immediately, then resume waiting without another user prompt.

Leave the monitor session untouched. Inspect confirmed and closed transitions in
a separate temporary command session with:

```bash
python -m experiment.degradation.cli show \
  --kind events --event all --phase both \
  --state-dir "$HOME/.local/state/rl-insight/degradation/default"
```

Inspect an `Updated event` notification with:

```bash
python -m experiment.degradation.cli show \
  --kind events --event all --phase latest \
  --state-dir "$HOME/.local/state/rl-insight/degradation/default"
```

After reporting, close only the temporary command session and resume polling the
original monitor session. Never return a final answer after an event report.

Match exactly one stored event using the logged target metric plus transition
step: `confirmed_at_step` for confirmed, `latest_analyzed_step` for updated, and
`closed_at_step` for closed. Read only the matching phase. If zero or multiple
events match, report the ambiguity and do not generate a diagnosis. A latest
refresh overwrites the same event's previous `latest` association; it never
creates a duplicate event or changes the original `confirmed` snapshot.

Generate an anomaly report only for `status=ok` with a selected event. For
`no_abnormal_events`, report that no event exists. For `uninitialized`, report
that no baseline exists. For `invalid_state` or command failure, report the
problem and use [troubleshooting.md](references/troubleshooting.md); do not
generate a fault hypothesis.

### Association output

Use the default `--top-k 25`; do not change it unless the user explicitly asks.
Render every returned entry in the complete stored Top-25, or every returned
entry when fewer than 25 are available. Group entries by the English metric
category, sort categories by their highest association score in descending
order, and sort metrics within each category by association score in descending
order. Break score ties by the stored `global_rank`. Keep every category in one
contiguous block. Write the category name only in the first row of that block
and leave the category cell blank in its remaining rows, even when the category
contributes many Top-25 metrics. Do not repeat the category name and do not
insert separator rows or horizontal rules between metrics or category blocks.
Category member count is not a fault decision.

The displayed table must contain only `Metric category`, `Metric name`, and
`Association score`. Display stored `association_percent` in the
`Association score` column. Use a compact Markdown table with ordinary pipe and
hyphen separators.
Do not add labels, direction, rank, correlation, random-forest fields, or
commentary inside this table. The complete deterministic event record remains
saved at
`<absolute-state-dir>/abnormal_data.json`; the formatted table is presented in
the conversation and is not a separate persisted file.

Illustrative format (the live report must include all returned Top-25 entries):

```text
Abnormal target metric: rl_insight_monitor_timing_s_step
Event phase: confirmed|latest|closed
Saved result: /absolute/path/to/degradation-state/abnormal_data.json
```

| Metric category | Metric name | Association score |
|---|---|---:|
| transfer_queue | tq_partition_consumption_progress | 94.80% |
|  | tq_storage_utilization_ratio | 91.25% |
|  | tq_storage_request_latency_p99 | 89.10% |
| latency | rl_insight_monitor_perf_throughput | 88.60% |
|  | rl_insight_monitor_perf_time_per_step | 84.30% |
| hardware_resources | rl_insight_monitor_perf_mfu_actor | 81.40% |

Treat association score as relative evidence, never fault probability, causal
contribution, or degradation magnitude.

For each valid transition, present one complete report in this order: abnormal
target, event phase, saved JSON path, compact Markdown association table, then
the root-cause analysis from Section 4.

If `monitor` exits non-zero, immediately report its exit status and stderr. Run
read-only `check` and local `show`, read
[troubleshooting.md](references/troubleshooting.md), and provide the likely
execution cause and next action. Do not reset state, change thresholds, or
generate an event diagnosis from a failed command.

## 4. Root Cause Analysis

### Input

- The uniquely matched abnormal target and event phase.
- The complete `top_k_by_category` result: metric category, metric name, global
  rank, and final `association_percent`.
- [diagnostic-experience.md](references/diagnostic-experience.md) as an
  incomplete, fallible prior.

### Analysis Capabilities

- Interpret metric semantics and the concentration of high association scores
  across categories.
- Use only the returned final association scores for evidence strength. Do not
  reopen raw series, query extra samples, write an additional analysis script,
  compare step values, or independently infer increases, decreases, trends, or
  change magnitude.
- Do not use candidate direction, point state (`NORMAL`, `UP`, `DOWN`, or
  `BETWEEN_MODES`), matched mode, or candidate abnormality as a diagnosis gate.
- When length or sequence metrics occupy a substantial part of the returned
  Top-K, `Sequence-length anomaly` MUST appear among the likely causes. When
  they also form the leading high-scoring evidence group, it MUST be cause 1.
  Apply this rule regardless of candidate direction or point state; do not infer
  whether sequence length increased or decreased. Many weak scores alone do not
  justify high confidence. This length-family rule is the explicit exception to
  the general rule that category size alone is not a vote.
- Rank fault domains and specific causes without using category count as a vote.
- Never invent unobserved hardware, network, operating-system, profiler, or
  device-health signals.

Use these fault domains: `Compute`, `Network`, `Host CPU`, and `HBM`. Always rank
exactly two distinct domains: primary and secondary. Return three to five
specific causes in confidence order; item 1 is the primary cause. `Low`
confidence is allowed, but every cause needs a concise evidence or uncertainty
basis and must not contradict observed evidence. Use only the cause vocabulary
in [diagnostic-experience.md](references/diagnostic-experience.md).
`Sequence-length anomaly` is a workload/data condition rather than one of the
four fault domains. When the sequence-length rule applies, the two fault domains
are only infrastructure alternatives and must not displace that diagnosis; keep
them low-confidence unless stronger non-length association evidence exists.

Apply this diagnosis contract only when the selected phase contains at least one
association entry. If it contains none, report the stored association status and
reason, state that root-cause evidence is insufficient, and do not fabricate
domains or causes.

### Output

```text
Primary fault domain: <domain> (confidence: High|Medium|Low)
Secondary fault domain: <different domain> (confidence: High|Medium|Low) — <brief evidence basis>

Likely fault causes:
1. <specific cause> (primary; confidence: High|Medium|Low) — <brief evidence basis>
2. <specific cause> (confidence: High|Medium|Low) — <brief evidence basis>
3. <specific cause> (confidence: High|Medium|Low) — <brief evidence basis>
[4. <specific cause> (confidence: High|Medium|Low) — <brief evidence basis>]
[5. <specific cause> (confidence: High|Medium|Low) — <brief evidence basis>]

Reasoning basis: <two or three concise professional sentences covering the strongest evidence and material contradiction>
```

### Example

```text
Primary fault domain: Network (confidence: Medium)
Secondary fault domain: Compute (confidence: Low) — throughput and MFU metrics also have meaningful association scores.

Likely fault causes:
1. Parameter-plane network congestion (primary; confidence: Medium) — transfer-queue and communication-sensitive latency metrics hold the strongest association scores.
2. Parameter-plane NIC bandwidth limitation (confidence: Low) — the same score pattern is compatible, but no direct NIC counter is present.
3. AI Core overload (confidence: Low) — MFU and throughput metrics provide weaker association evidence, and direct Cube evidence is absent.

Reasoning basis: The strongest association scores are concentrated in transfer-queue and latency metrics, while length and sequence metrics do not rank strongly. This supports a network-domain hypothesis but cannot locate the constrained component.
```

Use this section only to interpret observed evidence. Do not provide operational
procedures, commands, configuration changes, parameter values, or steps for
reproducing faults.

## 5. Reset

Stop monitoring, then preview the exact targets:

```bash
python -m experiment.degradation.cli reset \
  --state-dir "$HOME/.local/state/rl-insight/degradation/default"
```

After explicit user confirmation, back up and remove only those state files:

```bash
python -m experiment.degradation.cli reset \
  --state-dir "$HOME/.local/state/rl-insight/degradation/default" --yes
```

Do not use the legacy `--reset` flag for this workflow: it deletes state and
immediately retrains. Safe `reset --yes` backs up every existing target before
deletion and leaves the state `uninitialized`.
