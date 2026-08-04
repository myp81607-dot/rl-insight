# Input and Output Contract

The public algorithm boundary contains only in-memory time series, policies and
structured results. Data acquisition, transport conversion and result storage
belong to the caller.

## TimeSeries

`TimeSeries` represents one scalar metric:

```python
TimeSeries(
    timestamps=[1.0, 2.0, 3.0],
    values=[10.0, 10.1, 9.9],
)
```

Contract rules:

- `timestamps` and `values` are aligned and have equal lengths;
- timestamps and values are numeric; booleans are not accepted as numbers;
- non-finite or invalid pairs are removed together;
- valid pairs are sorted by timestamp;
- duplicate timestamps keep the last valid value from the original input;
- timestamp units must be consistent within an analysis request.

The detector validates and normalizes the sequence defensively, but callers
should construct canonical `TimeSeries` objects before invoking it.

## DetectionInput

`DetectionInput` explicitly separates healthy baseline data from the window to
be analyzed:

```python
data: DetectionInput = {
    'standard': {
        'timing_s/step': TimeSeries(
            timestamps=[1.0, 2.0, 3.0],
            values=[1.00, 1.01, 0.99],
        ),
        'gpu_utilization': TimeSeries(
            timestamps=[1.0, 2.0, 3.0],
            values=[40.0, 41.0, 39.0],
        ),
    },
    'inference': {
        'timing_s/step': TimeSeries(
            timestamps=[10.0, 11.0, 12.0],
            values=[1.00, 1.50, 1.60],
        ),
        'gpu_utilization': TimeSeries(
            timestamps=[10.0, 11.0, 12.0],
            values=[42.0, 88.0, 92.0],
        ),
    },
}
```

`standard` is used to identify stable health modes and build KDE models.
`inference` is classified against those models. Phase membership is always
explicit and is never inferred from a source name or timestamp.

`metric_policies` may override any selected metric; omitted metrics use the
module's unchanged default policy. Association targets and candidates must be
present in the selected metric set. Policies are plain in-memory mappings
supplied by the caller.

## DetectionResult

`detect()` and `detect_dataset()` return a `DetectionResult`. It is a
JSON-compatible structure; returning it does not write files or publish data.

Top-level fields:

- `taskId`: caller-provided analysis identity;
- `states`: numeric state for each successfully evaluated metric;
- `results`: per-metric thresholds, point diagnostics and interval details;
- `abnormalTimeRange`: confirmed intervals grouped by metric;
- `metricErrors`: optional isolated input, policy or detection errors;
- `associationAnalysis`: optional event-level association ranking.

Metric states are:

```text
0  detection completed
1  healthy standard data is insufficient
2  inference data is insufficient
```

A completed state does not by itself mean that degradation exists. Callers
must inspect the metric's `abnormalTimeRange`.

Each formal abnormal interval contains:

- `startTime` and `endTime`: published interval boundaries;
- `duration`: duration before display-boundary padding;
- `totalPointCount` and `abnormalPointCount`;
- `abnormalRate`;
- `abnormalType`: `UP`, `DOWN` or `BOTH`;
- `validationDetail` and `maximumAllowedGap`.

`results[metric].currentAbnormalTimeRange` contains intervals found in the
current batch. `results[metric].abnormalTimeRange` contains intervals that also
passed history confirmation. The top-level `abnormalTimeRange` mirrors the
confirmed intervals for convenient consumption.

Thresholds and point diagnostics remain metric-local. Failure of one metric is
reported through `metricErrors` and does not invalidate completed results for
other metrics. Association analysis is optional post-processing; its failure
does not rewrite KDE states or intervals.

For the algorithm behind these fields, see [Algorithm](ALGORITHM.md). For the
association result schema, see [Association Analysis](ASSOCIATION_ANALYSIS.md).
