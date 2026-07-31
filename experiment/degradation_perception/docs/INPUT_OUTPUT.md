# Degradation Input and Output

## Production input

Production analysis receives business parameters through
`rl-insight degradation analyze/monitor` semantics:

- RL-Insight task identity: `project`, `experiment_name`, optional `worker` and
  `replica`;
- one or more logical-to-Prometheus metric mappings;
- optional advanced PromQL templates;
- optional explicit standard/inference time boundaries;
- independent degradation algorithm configuration.

The production path does not accept or require a workflow YAML and does not
contain a required Prometheus URL. The managed Prometheus query endpoint is
resolved through RL-Insight Server service discovery.

`--prometheus-url` is a development/test override only.

## Window input

The window source order is:

```text
explicit CLI/API boundaries
> task metadata start time plus current detection time
> baseline/inference duration defaults
```

The defaults are 1800 seconds for each window. Explicit boundaries accept Unix
epoch seconds or ISO-8601 text.

The current RL-Insight Server exposes service information but no public task
metadata endpoint. Programmatic callers may provide
`task_metadata={"start_time": ...}` and CLI callers may use `--task-start`.
The integration never reads private Server runtime files.

## Prometheus query contract

Every query uses `/api/v1/query_range` and must return:

```json
{
  "status": "success",
  "data": {
    "resultType": "matrix",
    "result": [
      {
        "metric": {
          "__name__": "rl_insight_monitor_timing_s_step",
          "project": "verl",
          "experiment_name": "run-a",
          "worker": "trainer_0"
        },
        "values": [
          [1710000000, "1.0"],
          [1710000010, "1.1"]
        ]
      }
    ]
  }
}
```

The final query must identify exactly one scalar series. Native histogram-only
responses are rejected. Counter and histogram metrics must be converted to a
scalar with PromQL.

`MetricQuerySpec` supports `{{metric}}` and `{{labels}}` placeholders so one
query template can reuse RL-Insight task labels for both phases.

## Internal algorithm input

After strict matrix conversion, the unchanged detector receives:

```json
{
  "standard": {
    "timing_s/step": {
      "timestamps": [1710000000, 1710000010],
      "values": [1.0, 1.1]
    }
  },
  "inference": {
    "timing_s/step": {
      "timestamps": [1710001800, 1710001810],
      "values": [1.3, 1.4]
    }
  }
}
```

Phase membership is explicit; it is never inferred from filenames.

The existing offline `main --path` contract remains available for tests and
local data. It accepts UTF-8 JSON whose root contains exactly `standard` and
`inference`. Each metric is either canonical `timestamps`/`values` data or one
already-fetched single-series Prometheus matrix.

## JSON output

`analyze` prints one complete, strict JSON-serializable `DetectionResponse`.
`monitor` prints one response per run as JSON Lines. The response includes:

- `taskId`;
- per-metric `states`;
- detailed `results`, including thresholds, point diagnostics and
  `abnormalTimeRange`;
- top-level `abnormalTimeRange`;
- isolated `metricErrors`;
- optional `associationAnalysis`.

When `--output-dir` is set, the latest run is also stored as:

```text
detection_response.json
query_diagnostics.json
```

## Prometheus output

`monitor` exposes low-cardinality derived gauges through a persistent
`/metrics` endpoint and registers that endpoint using the existing RL-Insight
scrape target API.

Prometheus receives status, threshold envelopes, abnormal rate/count, latest
detection timestamp, interval-active state and bounded association Top-K.

Complete intervals and complex diagnostics remain JSON-only. Query strings,
timestamps and interval payloads are never Prometheus labels.
