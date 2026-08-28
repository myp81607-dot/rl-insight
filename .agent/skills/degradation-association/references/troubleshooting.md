# Degradation troubleshooting

Use the core CLI's read-only `check` before `run-once` or `monitor`, or when
troubleshooting Prometheus/readiness. It reports the selected global step,
discovered metric counts, optional dependencies, and state-file health. Do not
require it before offline `show` or `reset`.

## Prometheus request fails

- The first attempt uses `http://127.0.0.1:9090`. If it is unreachable, ask the
  user whether the Prometheus URL and state directory should be changed. Do not
  guess another endpoint, state path, or task selector.
- Confirm the URL points to Prometheus itself, commonly port 9090, rather than a
  training service or model endpoint.
- Test the Prometheus readiness endpoint and API reachability from the same host
  or container that runs rl-insight.
- Use `--insecure` only for a known TLS certificate issue; do not use it for plain
  HTTP or as a general connectivity workaround.

## Global step returns zero or multiple series

- Confirm that `rl_insight_monitor_training_global_step` is being scraped.
- Narrow the training task at the Prometheus/job configuration level when
  multiple global-step series exist. The v0.1 instant query uses the metric name
  directly and requires uniqueness; `--series-selector` does not filter that
  instant query.
- Do not select an arbitrary series or merge steps from different tasks.

## No target metric is discovered

- Inspect the target names in `experiment/degradation/metrics.py` and compare them
  with the metrics emitted by the active trainer.
- Discovery defaults to all Prometheus series. If a selector was explicitly
  supplied, verify that it does not exclude the target and that the lookback
  interval overlaps recent samples.
- Do not promote arbitrary candidate metrics to targets merely to make preflight
  pass. Add a target only when its degradation direction and operational meaning
  are understood.

## No candidate metric is discovered

- Treat `error_kind: no_candidate_metrics` as association readiness failure and
  do not start `run-once` or `monitor`.
- Check the candidate catalog, metric export, and Prometheus lookback. If a
  selector was explicitly supplied, verify that it does not exclude candidates.
- Do not invent association evidence from target metrics alone.

## Monitoring stops after baseline training

- Start continuous work with `monitor`, not `run-once`. `run-once` intentionally
  exits after initialization and one poll.
- Keep the monitor process in a persistent terminal/session. Without a baseline,
  it waits for 30 complete new steps, saves the baseline at the next-step
  boundary, and then continues detection in the same process.
- Association runs automatically at confirmed and closed transitions. Keep the
  monitoring task active so the agent can inspect and report each transition.

## Monitor exits non-zero

- Preserve and report the original exit status and stderr before running any
  diagnostic command.
- Run read-only `check` and offline `show` independently. If either diagnostic
  also fails, report that failure without replacing the original monitor error.
- Use the relevant section below to explain the likely cause and next action.
- Automatically correct only an equivalent working directory, known existing
  virtual environment, absolute state path, or command syntax that does not
  change values. Retry the same read-only diagnostic at most once.
- For an explicit timeout, connection reset, or HTTP 5xx only, confirm the old
  monitor exited and no duplicate uses the state directory, then retry monitor
  once with identical arguments. Never loop.
- Ask before installing dependencies, changing URL/selector/metrics/thresholds,
  using `--insecure`, resetting or editing state, killing a live process, editing
  code, or starting another monitor. Do not generate an event fault hypothesis.

## Baseline cannot be trained

- Ensure at least 30 complete consecutive steps are available with the default
  configuration, including the next step timestamp used as the final boundary.
- Inspect skipped-series reasons for fewer than 20 finite values, KDE failure, or
  absence of a stable segment.
- Avoid lowering sample or stability thresholds without documenting how the
  weaker baseline will be validated.

## Existing baseline does not match overrides

- A saved baseline is frozen. Baseline step count, ratio, alpha, or minimum-sample
  overrides require either matching stored parameters or an explicit reset.
- Stop monitoring and preview reset first. With `--yes`, the CLI backs up each
  existing state file before removing it.

## Confirmed event has no useful association

- Correlation can run with pairwise-aligned values even when random forest cannot.
- Forest analysis needs at least one abnormal candidate, sufficient common steps,
  and both target classes in chronological training and validation partitions.
- A zero or unavailable forest importance does not invalidate a valid correlation
  result. Report each evidence path and its reason separately.

## State display fails

- `uninitialized` means both files are absent before initialization or after
  reset. Run initialization when appropriate.
- `no_abnormal_events` means the baseline is valid but no event has been stored;
  it is a successful result, not a failure.
- `ok` means a valid baseline and at least one selectable event exist.
- `invalid_state` covers corrupt JSON, an unsupported schema, a non-regular state
  path, or an abnormal file without a baseline. Restore a known backup or reset
  with explicit approval; do not hand-edit live state while monitoring is running.
- Only `invalid_state` is a state-level exit `1`. Connectivity, dependency,
  argument, and explicit phase-selection failures are separate command errors.
- Use a separate `--state-dir` per independent training task or experiment.
- If the intended absolute state directory cannot be established, ask whether
  it and the Prometheus URL should be changed; do not silently create or select
  another task's state directory.

## Python import fails

Install the degradation extra described in the Prepare section of `SKILL.md`.

Locate the source checkout containing `experiment/degradation/cli.py`, use it as
the working directory, and pass an absolute `--state-dir`. The `experiment`
package is not installed in the published wheel.
