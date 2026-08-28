---
name: degradation-association
description: "Continuously operate rl-insight's Prometheus-backed degradation experiment: train or load a baseline, report every confirmed/closed target transition, preserve the grouped Top-25 evidence, and infer ranked fault domains and causes."
user_invocable: true
---

# Degradation association

Use the deterministic implementation in `experiment/degradation`. Interpret its
evidence; do not reimplement its algorithms in the skill.

## Prepare

1. Locate the rl-insight checkout containing `experiment/degradation/cli.py` and
   run every command with that checkout as the working directory. The experiment
   is source-tree code and is not installed in the wheel.
2. Pass an absolute `--state-dir`, especially after changing the working
   directory. Keep one state directory per training task.
3. Verify the optional runtime before detection. If it is missing, report it and
   obtain approval before running `pip install -e ".[degradation]"`.
4. Start with the default Prometheus URL `http://127.0.0.1:9090`. Reuse the
   established absolute state directory for this training task. If the default
   URL is unreachable or the state directory cannot be established, ask the
   user whether the Prometheus URL and state directory should be changed. Do
   not guess an alternate URL, path, or task label.
5. Omit `--series-selector` by default so discovery uses all Prometheus series.
   Never invent a trainer/job selector. Use a narrower selector only when the
   user explicitly supplies or requests one.

Read [algorithm-contract.md](references/algorithm-contract.md) before changing
parameters. Read [troubleshooting.md](references/troubleshooting.md) after a
failed command or `invalid_state`.

## Operate

Run the read-only `check` before starting detection:

```bash
python -m experiment.degradation.cli check \
  --prometheus-url http://127.0.0.1:9090 \
  --state-dir /absolute/path/to/degradation-state
```

Confirm one integer-valued global-step series, at least one configured target,
at least one configured candidate, the required dependencies, and valid state.
If `check` reports `error_kind=no_candidate_metrics`, stop; do not start
`run-once` or `monitor`. Check the selector, metric export, and Prometheus data.
Do not require `check` before offline `show` or `reset`; they operate only on
persisted local state and must remain usable when Prometheus is unavailable.

On a bare invocation of this skill, or when the user asks to start monitoring,
run `monitor` directly as a foreground process inside one persistent
terminal/session:

```bash
python -m experiment.degradation.cli monitor \
  --prometheus-url http://127.0.0.1:9090 \
  --state-dir /absolute/path/to/degradation-state
```

Do not run `run-once` first and do not wait for another user prompt to start
`monitor`. If no baseline exists, this same process collects 30 complete new
steps, writes the baseline after observing the next-step boundary, and then
continues detection without exiting. If a baseline exists, it loads it and
continues detection immediately. Keep the task attached to that same session,
poll its output repeatedly, and keep it alive until the user asks to stop. Do
not detach, daemonize, interrupt, replace, or prematurely restart a live monitor.

Do not finish the task after starting `monitor`. Keep waiting on the same
terminal/session until the user explicitly asks to stop. Do not return a final
answer after startup or baseline completion.

When the session logs `Trained <count> baselines from steps <start>-<end>;
skipped <count> series`, immediately notify the user:

```text
Baseline ready: trained <count> baselines from steps <start>-<end>; skipped <count> series. Monitoring continues in the same process.
```

This notification is not a completion signal. Do not stop, close, interrupt,
restart, or replace the monitor process after sending it; continue watching the
same session for confirmed and closed events.

Omitting `--series-selector` uses the core default `{__name__=~".+"}`. This
discovers all Prometheus series, after which the runtime uses every discovered
configured target and candidate. Do not silently narrow discovery to
`job="training"`, trainer metrics, or any other guessed label set. Association
analysis is enabled by default and runs automatically for every confirmed and
closed target event.

Use `run-once` only when the user explicitly requests a one-time validation or
single poll:

```bash
python -m experiment.degradation.cli run-once \
  --prometheus-url http://127.0.0.1:9090 \
  --state-dir /absolute/path/to/degradation-state
```

Keep watching the persistent monitor session. On every `Confirmed event` or
`Closed event` transition, immediately run `show --event all --phase both`,
then match exactly one stored event by the logged target metric and transition
step: use `confirmed_at_step` for confirmed and `closed_at_step` for closed. Read
only that event's matching phase. If zero or multiple events match, report the
ambiguity and do not guess or generate a fault hypothesis. Otherwise report the
abnormal information, considered fault type, and association evidence to the
user. Do not wait for another inspection request. A detached process without an
active wait or monitoring task is insufficient because it cannot deliver this
diagnosis.

If `monitor` exits non-zero, immediately report its exact exit status and stderr.
Run read-only `check` and offline `show` as independent diagnostics, read the
matching section of [troubleshooting.md](references/troubleshooting.md), and
report the likely cause and next action. Do not generate an event fault
hypothesis from a command failure.

Automatically correct only an equivalent working directory, known existing
environment, absolute state path, or command syntax, then retry a read-only
command once. A transient Prometheus timeout, connection reset, or HTTP 5xx may
restart `monitor` once with identical arguments only after the old process exits.
Read [troubleshooting.md](references/troubleshooting.md) for the full boundary;
anything that changes dependencies, configuration, state, code, or a live
process requires approval.

## Inspect and diagnose

Inspect the latest event and best available phase:

```bash
python -m experiment.degradation.cli show \
  --kind all --event latest --phase auto \
  --state-dir /absolute/path/to/degradation-state
```

`latest` selects the last stored, most recently confirmed event. `auto` selects
`closed` when present and otherwise `confirmed`. Use `--event all` or `--phase
confirmed|closed|both` only when another view is needed. An unavailable explicit
phase is a command error.

Interpret these state statuses:

- `ok`: valid baseline and at least one abnormal event.
- `no_abnormal_events`: valid baseline and no abnormal event; this is success.
- `uninitialized`: neither state file exists, including after reset.
- `invalid_state`: corrupt, unsupported, non-regular, or inconsistent state.

The first three exit with `0`; `invalid_state` exits with `1`. Structured
`check`, `show`, and `reset` execution or selection failures return `{"ok":
false, "error": "..."}` on stderr and do not create another state status.
`run-once` and `monitor` retain the runtime's normal stderr and exit codes.

Generate the diagnostic summary only when `show` returns `status=ok` with at
least one selected event. For `no_abnormal_events`, report that no abnormal
event exists. For `uninitialized`, report that no baseline exists. For
`invalid_state`, read [troubleshooting.md](references/troubleshooting.md) and
report the state problem. Never generate a fault hypothesis for these statuses
or for a command failure.

For a valid selected event, read `top_k_by_category` as grouped presentation of
the stored flat Top-K.
Categories follow their best original `global_rank`; metric count is context, not
a fault decision. Use [diagnostic-experience.md](references/diagnostic-experience.md)
only as an incomplete, fallible prior. Combine the model's own technical
knowledge with metric meaning, score, direction, correlation/random-forest
availability, labels, phase, temporal context, category evidence, and
contradictions. Category size and experience examples must not decide the result.

For every valid event phase, rank these fault domains:

- `Compute`
- `Network`
- `Host CPU`
- `HBM`

Return the two best-supported, distinct domains in order and three or four
ranked specific causes from the vocabulary in
[diagnostic-experience.md](references/diagnostic-experience.md); do not add new
domain or cause labels.
The second domain may have `Low` confidence, but do not omit it or assign fake
numeric probabilities. These are diagnostic hypotheses, not claims that an
unobserved hardware, network, or operating-system signal was measured.

Start each confirmed or closed transition report with exactly:

```text
Abnormal target metric: <metric and identifying labels>
Event phase: confirmed|closed
Most likely fault domain: Compute|Network|Host CPU|HBM (confidence: High|Medium|Low)
Second likely fault domain: Compute|Network|Host CPU|HBM (confidence: High|Medium|Low)
Likely fault causes:
1. <specific cause> — <brief evidence basis>
2. <specific cause> — <brief evidence basis>
3. <specific cause> — <brief evidence basis>
[4. <specific cause> — <brief evidence basis>]
Reasoning basis: <two or three concise professional sentences>
```

Then print `Association evidence categories:` followed by each non-empty English
category block and every metric in the stored Top-25, or every returned entry
when fewer than 25 are available. Preserve `global_rank` order and include
identifying labels, association percentage, direction, and rank for every item;
never shorten the block to a few representative metrics.
Do not translate category names, metric names, domains, or causes. Treat
association percentage as relative evidence, not fault probability, causal
contribution, or degradation magnitude.

## Reset safely

Stop monitoring and preview the exact targets:

```bash
python -m experiment.degradation.cli reset \
  --state-dir /absolute/path/to/degradation-state
```

After explicit user confirmation, back up and remove only those state files:

```bash
python -m experiment.degradation.cli reset \
  --state-dir /absolute/path/to/degradation-state --yes
```

Do not use the legacy `--reset` flag for this workflow: it deletes state and
immediately retrains. Safe `reset --yes` backs up all existing targets before any
deletion and leaves the state `uninitialized`.

## Preserve scope

- Preserve KDE fitting, multimodal ranges, the 3-of-5 lifecycle rule, and the
  correlation/random-forest ranking unless explicitly asked to change them.
- Keep global step outside target/candidate modeling and raw Histogram/Counter
  families outside v0.1 KDE.
- Do not claim multi-task isolation, retry orchestration, daemon management,
  alert delivery, or Grafana administration.
- Report the command, selector, state directory, exit status, and material
  warnings after each operation.
