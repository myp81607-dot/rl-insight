# Degradation Perception Experiment

This experiment detects performance degradation in RL training metrics stored in
Prometheus. It can train or load a KDE baseline, monitor newly completed training
steps, confirm target degradation, rank associated metrics, and persist the results
as JSON.

The current implementation is a source-tree v0.1 experiment. It assumes one active
training task and one unique `rl_insight_monitor_training_global_step` series.

## Workflow

```text
Start CLI
  -> load standard_data.json when it exists
  -> otherwise collect 40 complete training steps
  -> align metric samples by global step
  -> train and save KDE baselines
  -> poll Prometheus every 5 minutes by default
  -> classify new target and candidate points
  -> confirm a target event with the 3-of-5 rule
  -> run correlation and random-forest association analysis
  -> save the default Top-5 to abnormal_data.json
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
    ├── storage.py
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
| `storage.py` | Reads and writes baseline and abnormal-event JSON files. |
| `runtime.py` | Orchestrates training, polling, detection, association, and persistence. |
| `cli.py` | Parses one-command runtime, inspection, and reset operations. |

The detailed algorithm contract is documented in
[`docs/algorithm.md`](../docs/algorithm.md).

## Run from Source

Install the optional dependencies and run commands from the repository root:

```bash
pip install -e ".[degradation]"
```

Start automatic baseline training or load an existing baseline, then continue
polling:

```bash
python -m experiment.degradation.cli \
  --prometheus-url http://127.0.0.1:9090 \
  --series-selector '{job="training"}'
```

Use a selector that identifies the current training task. The global-step instant
query must still return exactly one series.

Common operations:

```bash
# Initialize and perform only one detection poll.
python -m experiment.degradation.cli --once

# Print stored abnormal events and their correlation/RF components.
python -m experiment.degradation.cli --show-components

# Delete both state files and train a new baseline.
python -m experiment.degradation.cli --reset

# Reset and override baseline and runtime parameters.
python -m experiment.degradation.cli \
  --reset \
  --ratio 1.6 \
  --baseline-alpha 0.05 \
  --minimum-samples 20 \
  --poll-interval 300 \
  --top-k 5
```

Important CLI options:

| Option | Purpose | Default |
|---|---|---:|
| `--baseline-steps` | Complete steps used for a new baseline. | `40` |
| `--minimum-samples` | Required non-missing values per series. | `20` |
| `--poll-interval` | Seconds between detection polls. | `300` |
| `--query-step` | Prometheus range-query resolution in seconds. | `10` |
| `--ratio` | Sets both KDE baseline expansion ratios. | `1.05` |
| `--pre-context-steps` | Steps retained before an event for association. | `30` |
| `--top-k` | Associated metrics saved for each event phase. | `5` |
| `--target-metric` | Adds another UP-policy target; repeat as needed. | catalog targets |
| `--state-dir` | Directory containing the two JSON state files. | current directory |
| `--reset` | Clears saved baseline and event history before training. | disabled |
| `--once` | Runs one poll instead of continuous monitoring. | disabled |
| `--show-components` | Prints `abnormal_data.json` without Prometheus. | disabled |

Run `python -m experiment.degradation.cli --help` for the complete option list.

## Output Files

- `standard_data.json` stores fitted normal ranges and baseline parameters. It does
  not store the original 40-step observations.
- `abnormal_data.json` stores confirmed and closed events, target direction, Top-K
  candidate direction, association percentage, correlation values, and
  random-forest components.

Metric names that are not exposed by the current training task are skipped. Raw
Histogram targets and raw Counter candidates remain catalog entries but are not
sent directly to KDE in v0.1.

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
and CLI result inspection using deterministic data and fake Prometheus clients.

Optional static checks:

```bash
python -m ruff check experiment/degradation tests/monitor/ut/degradation
python -m ruff format --check experiment/degradation tests/monitor/ut/degradation
python -m mypy experiment/degradation
```

## v0.1 Scope

- Baselines survive restart; active detector windows, event context, and polling
  cursors do not.
- Prometheus request retry, multi-task isolation, dynamic series discovery, and
  alert delivery are outside the current experiment.
- This directory is not included in the published wheel or the root `rl-insight`
  command yet. Run it from the repository source tree.
- A real-server smoke test should use task-specific labels and confirm that every
  integer global step is available at the selected query resolution.
