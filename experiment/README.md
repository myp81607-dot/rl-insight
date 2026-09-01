# Degradation Perception Experiment

This experiment detects performance degradation in RL training metrics stored in
Prometheus. It can train or load a KDE baseline, monitor newly completed training
steps, confirm target degradation, rank associated metrics, and persist the results
as JSON.

The current implementation is a source-tree experiment. It assumes one active
training task and one unique `rl_insight_monitor_training_global_step` series.

## Workflow

```text
Start CLI
  -> load standard_data.json when it exists
  -> otherwise collect 30 complete training steps
  -> align metric samples by global step
  -> train and save KDE baselines
  -> poll Prometheus every 5 minutes by default
  -> classify new target and candidate points
  -> confirm a target event with the 3-of-5 rule
  -> run correlation and random-forest association analysis
  -> save the default Top-25 to abnormal_data.json
```

Association runs once when an event is confirmed and again when the same event is
closed.

## Layout

```text
experiment/
├── README.md
└── degradation/
    ├── __init__.py
    ├── metrics.py
    ├── config.py
    ├── prometheus.py
    ├── window.py
    ├── baseline.py
    ├── detector.py
    ├── association.py
    ├── state_schema.py
    ├── storage.py
    ├── presentation.py
    ├── runtime.py
    └── cli.py
```

| File | Responsibility |
|---|---|
| `metrics.py` | Defines the global-step ruler and default target/candidate catalog. |
| `config.py` | Validates typed runtime and algorithm parameters. |
| `prometheus.py` | Calls the Prometheus HTTP API and parses samples into `TimeSeries`. |
| `window.py` | Aligns metric samples into step-indexed `StepFrame` objects. |
| `baseline.py` | Fits KDE baselines and records stable normal ranges. |
| `detector.py` | Classifies points and tracks target events with the 3-of-5 rule. |
| `association.py` | Ranks candidates using correlation and random-forest evidence. |
| `state_schema.py` | Validates and classifies the two-file persisted-state contract. |
| `storage.py` | Reads, writes, inspects, backs up, and deletes configured state files. |
| `presentation.py` | Selects events/phases and groups Top-K evidence for model use. |
| `runtime.py` | Orchestrates training, polling, detection, association, and persistence. |
| `cli.py` | Exposes runtime, readiness, inspection, and safe-reset commands. |

The detailed algorithm contract is documented in
[`docs/algorithm.md`](../docs/algorithm.md).

## Run from Source

Install the optional dependencies and run commands from the repository root:

```bash
pip install -e ".[degradation]"
```

Start monitoring directly. With no saved baseline, the same process collects and
fits 30 complete new steps and then continues detection; with a saved baseline,
it loads it and begins detection immediately:

```bash
python -m experiment.degradation.cli monitor \
  --prometheus-url http://127.0.0.1:9090 \
  --state-dir "$HOME/.local/state/rl-insight/degradation/default"
```

Do not use `run-once` to bootstrap continuous monitoring. `monitor` performs
baseline initialization and remains running until stopped. The CLI is a
foreground process; an agent that must keep responding while it runs should keep
it attached to one persistent terminal/session rather than starting a second
replacement monitor process after baseline training.

After starting `monitor`, the agent should keep the task active and continue
waiting on the same terminal/session until the user explicitly asks to stop. It
should not return a final answer after startup or baseline completion.

After the runtime logs `Trained ... baselines from steps ...`, an agent should
notify the user that the baseline is ready and that monitoring continues. It
must keep the same monitor process running; the notification is not permission
to stop, restart, or replace it.

Omitting `--series-selector` uses `{__name__=~".+"}` and discovers all Prometheus
series. The runtime then uses every discovered metric in the configured target
and candidate catalogs. Supply a narrower selector only intentionally; do not
guess a trainer/job label. The global-step instant query must still return
exactly one series.

An agent should first use the default Prometheus URL
`http://127.0.0.1:9090` and the absolute expansion of
`~/.local/state/rl-insight/degradation/default`. If either is unavailable, it
should ask whether the Prometheus URL and state directory need to be changed
instead of guessing an alternate URL, path, or label selector. Another training
task requires a separate state directory.

Common operations:

```bash
# Read-only Prometheus, dependency, metric, and state preflight.
python -m experiment.degradation.cli check \
  --state-dir "$HOME/.local/state/rl-insight/degradation/default"

# Initialize and perform only one detection poll.
python -m experiment.degradation.cli run-once \
  --state-dir "$HOME/.local/state/rl-insight/degradation/default"

# Validate state and show category-grouped evidence.
python -m experiment.degradation.cli show \
  --event latest --phase auto \
  --state-dir "$HOME/.local/state/rl-insight/degradation/default"

# Preview, then explicitly confirm a backup-first reset.
python -m experiment.degradation.cli reset \
  --state-dir "$HOME/.local/state/rl-insight/degradation/default"
python -m experiment.degradation.cli reset \
  --state-dir "$HOME/.local/state/rl-insight/degradation/default" --yes
```

Run `check` before `run-once` or `monitor`. It requires at least one configured
target and candidate metric; zero candidates returns `ok: false`,
`error_kind: no_candidate_metrics`, and a non-zero exit status. Do not require
`check` before `show` or `reset`: those commands inspect persisted local state
and remain usable when Prometheus is unavailable. Do not start `run-once` or
`monitor` until the selector, metric export, or Prometheus data exposes a
configured candidate.

Baseline fitting must retain at least one configured target and one configured
candidate. Otherwise initialization exits instead of starting a monitor that
cannot produce target events or association evidence.

Correlation and random-forest association are always active. Runtime analyzes
and persists evidence at both confirmed and closed transitions. An agent watching
the monitor should leave its session untouched, run `show --event all --phase
both` in a separate temporary command session, and report exactly one event
matched by target metric plus `confirmed_at_step` or `closed_at_step`. It then
closes only the temporary session and resumes polling the original monitor.
Confirmed and closed are separate reports for each target transition, not
reports for each associated candidate metric. An event report is not a final
answer.

For a valid unique event with association evidence, the agent ranks exactly two
distinct fault domains from `Compute`, `Network`, `Host CPU`, and `HBM`, then
returns three to five specific causes in confidence order. Cause 1 is primary;
every cause needs an evidence or uncertainty basis and must not contradict the
observed evidence. A selected phase with no association entries reports
insufficient root-cause evidence instead of fabricating a ranking.

The complete stored Top-25 (or every returned entry when fewer than 25 are
available) is rendered as a compact Markdown table. Categories are ordered by
their highest association score, and metrics within each category are ordered by
score; score ties retain stored global-rank order. The displayed table contains
only the metric category, metric name, and association score. The underlying
JSON retains labels, direction, rank, correlation, and random-forest fields for
analysis. Association percentage is not fault probability. Agents use the
default `--top-k 25` unless the user explicitly requests another value.

Diagnostic experience is an incomplete, fallible prior. The agent must combine
its own technical knowledge with metric semantics, scores, direction, labels,
correlation/random-forest evidence, phase, temporal context, and contradictions.
Category count or an experience example alone must not decide the domains or
causes, and absent direct hardware/network/OS signals must not be presented as
observed facts. When event matching is not unique, report ambiguity and do not
diagnose.

If monitoring exits non-zero, preserve and report the original status and stderr,
then run read-only `check` and local `show` independently. Report the likely
execution cause and next action. Do not automatically reset state, change
thresholds, or restart monitoring; detailed boundaries live in the Skill
troubleshooting reference.

The source-tree CLI still accepts the original flat flags for compatibility:
`--once` runs one poll, `--show-components` prints raw `abnormal_data.json`, and
`--reset` deletes state and immediately retrains. These legacy operations retain
their original semantics; use the subcommand interface for model-facing
inspection and backup-first reset.

Important CLI options:

| Option | Purpose | Default |
|---|---|---:|
| `--baseline-steps` | Complete steps used for a new baseline. | `30` |
| `--minimum-samples` | Required non-missing values per series. | `20` |
| `--poll-interval` | Seconds between detection polls. | `300` |
| `--query-step` | Prometheus range-query resolution in seconds. | `10` |
| `--ratio` | Sets both KDE baseline expansion ratios. | `1.05` |
| `--pre-context-steps` | Steps retained before an event for association. | `30` |
| `--top-k` | Associated metrics saved for each event phase. | `25` |
| `--target-metric` | Adds another UP-policy target; repeat as needed. | catalog targets |
| `--state-dir` | Directory containing the two JSON state files. | current directory for the core CLI; Skill passes the explicit path above |
| legacy `--reset` | Clears state and immediately retrains. | disabled |
| legacy `--once` | Runs one poll instead of continuous monitoring. | disabled |
| legacy `--show-components` | Prints raw `abnormal_data.json`. | disabled |

Run `python -m experiment.degradation.cli <subcommand> --help` for each new
operation, or omit the subcommand to view the compatible legacy option list.

## Inspect with the Core CLI

`show` defaults to the latest stored event and `auto` phase: it selects `closed`
when available and otherwise `confirmed`. `--event latest|all` and `--phase
auto|confirmed|closed|both` provide explicit alternatives.

The CLI groups the stored flat Top-K under `top_k_by_category` without changing
either JSON file. The agent renders the grouped result as a score-descending,
compact Markdown table containing only category, metric name, and association
score; metric count is context, not a mechanical fault decision. `show` reports
`ok`,
`no_abnormal_events`, `uninitialized`, or `invalid_state`. The first three exit
with `0`; `invalid_state` exits with `1`. Selection and execution failures return
a separate error response for structured `check`, `show`, and `reset` commands
and do not create another state status. Runtime commands retain normal stderr and
exit codes.

Generate a diagnostic summary only for `status: ok` with a selected event. For
`no_abnormal_events`, `uninitialized`, `invalid_state`, or command failure,
report the status or error and next action without generating a fault hypothesis.

Safe `reset` previews exact files. Stop monitoring, then add `--yes` only after
confirmation. It backs up every existing target before deletion; the next `show`
is `uninitialized`. Legacy `--reset` does not provide this flow.

## Output Files

- `<state-dir>/standard_data.json` stores fitted normal ranges and baseline
  parameters. It does not store the original 30-step observations.
- `<state-dir>/abnormal_data.json` stores confirmed and closed events, target
  direction, Top-K candidate direction, association percentage, correlation
  values, and random-forest components.

The compact Markdown association table and root-cause narrative are conversation
output; the CLI does not persist a separate formatted report file.

Metric names that are not exposed by the current training task are skipped. Raw
Histogram targets and raw Counter candidates remain catalog entries but are not
sent directly to the current KDE pipeline.

Detected degradation is stored as an event. Runtime failures are written to stderr,
and the CLI exits with status `1`.

## Tests

Run the CLI tests:

```bash
python -m pytest tests/monitor/ut/degradation/test_cli.py -q
```

Run the complete degradation test suite:

```bash
python -m pytest tests/monitor/ut/degradation -q
```

The suite covers Prometheus response parsing, step alignment, KDE fitting, detector
events, association ranking, JSON persistence, runtime initialization, reset order,
CLI result inspection, skill state semantics, event/phase selection, grouped
presentation, and safe reset using deterministic data and fake Prometheus clients.

Optional static checks:

```bash
python -m ruff check experiment/degradation tests/monitor/ut/degradation
python -m ruff format --check experiment/degradation tests/monitor/ut/degradation
python -m mypy experiment/degradation
```

## Scope

- Baselines survive restart; active detector windows, event context, and polling
  cursors do not.
- Prometheus request retry, multi-task isolation, dynamic series discovery, and
  alert delivery are outside the current experiment.
- This directory is not included in the published wheel or the root `rl-insight`
  command yet. Run it from the repository source tree.
- A real-server smoke test should use task-specific labels and confirm that every
  integer global step is available at the selected query resolution.
