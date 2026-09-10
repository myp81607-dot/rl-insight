# Offline degradation association

Analyze a user-selected time range directly from RL-Insight's local Prometheus
TSDB. The first step value visible after the selected start is skipped as
potentially partial. The next 30 complete steps train a baseline when one is not
supplied; later steps are checked for target events and Top-25 associations.

## Install

```bash
pip install -e ".[degradation]"
```

## Analyze a range

```bash
python -m experiment.degradation.cli analyze \
  --start-time 2026-09-08T09:00:00+08:00 \
  --end-time 2026-09-08T12:00:00+08:00
```

The default TSDB is `~/.rl-insight/data/prometheus`. Use `--data-dir`,
`--promtool`, `--analysis-dir`, or `--baseline-file` to override the respective
paths.

The CLI writes deterministic baseline, event, and analysis JSON. The repository
Skill at `.agent/skills/degradation-association-offline/SKILL.md` turns the final
association evidence into the grouped Markdown report and root-cause summary.
