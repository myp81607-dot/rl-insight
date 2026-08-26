# Diagnostic experience

These incomplete, fallible cases are weak priors only. Always combine the
model's own technical knowledge with the actual grouped evidence.

The four fault labels below are this skill's diagnostic vocabulary, not four
official Huawei fault-class names. Here, `AI Core overload` means a Cube/matrix
compute-side bottleneck, while `AI Vector overload` means a Vector compute-side
bottleneck.

## Distinguishing evidence

- `AI Core overload`: prefer this when Cube-side evidence dominates, such as
  sustained `aic_time`, `aic_total_cycles`, `aic_cube_ratio`, `mac`, or Cube
  FLOPs together with compute latency or throughput degradation. Stable
  frequency and healthy device/core evidence strengthen this diagnosis. A
  dominant Vector ratio argues against it.
- `AI Vector overload`: prefer this when Vector-side evidence dominates, such
  as sustained `aiv_time`, `aiv_total_cycles`, `aiv_vec_ratio`, `vec`, or
  Vector FLOPs together with affected latency or throughput. An `aivec error`
  tied to a core ID is Vector-specific evidence, but an exception may indicate
  a software or hardware fault rather than overload. Dominant Cube work argues
  against this diagnosis.
- `NPU frequency throttling`: prefer this when current AI Core frequency falls
  below its normal or rated level while AI tasks remain active, and the drop is
  temporally coherent with performance loss and temperature, power, or
  overtemperature evidence. A low frequency while no AI task is running can be
  a normal low-power state and is not sufficient evidence.
- `NPU core offline`: prefer this for abrupt, severe, and persistent capacity
  loss accompanied by a disappeared core/device/rank, zero or flat activity
  while peers remain active, repeated errors on the same chip/core ID, ECC/RAS
  events, unhealthy device state, or abnormal AI Core diagnosis/stress results.
  If degradation is exceptionally severe and is strongly associated with an
  NPU frequency reduction, elevate this diagnosis above ordinary throttling
  when the loss persists or any independent offline/hardware signal is present.
  Frequency reduction alone still supports throttling, not core offline.

Direct hardware and profiler signals outrank indirect category patterns. When
they are absent, use coherent latency, `hardware_resources`, `vllm_engine`,
`rollout_quality`, and `transfer_queue` evidence only as indirect support and
lower confidence rather than inventing a hardware observation.

These are not category-to-fault mappings. Do not decide by category count or an
experience example alone. Weigh metric semantics, association strength,
direction, labels, correlation/random-forest availability, temporal coherence,
event phase, supporting evidence, and contradictions.

Return exactly one of the four fault labels. If evidence is weak or mixed, choose
the best-supported label with `Low` confidence and state the limitation without
listing an alternative diagnosis.

## Official Ascend sources

- [ArithmeticUtilization field definitions](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0093.html)
  distinguish AI Cube Core and AI Vector Core execution time, cycles, ratios,
  and FLOPs.
- [PipeUtilization field definitions](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0099.html)
  distinguish Cube, Vector, scalar, and memory-transfer pipeline time and ratios.
- [AI Core frequency viewing](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/profiling/atlasprofiling_16_0059.html)
  explains that frequency reduction degrades active-task performance, can be
  temperature protection, and can also be a normal idle low-power state.
- [AI Core error symptoms](https://www.hiascend.com/document/detail/en/canncommercial/850/maintenref/troubleshooting/troubleshooting_0004.html)
  identifies `aivec`/`aicore` exceptions and the diagnostic value of chip, die,
  and core IDs.
- [AI Core error locating](https://www.hiascend.com/document/detail/en/canncommercial/800/maintenref/troubleshooting/troubleshooting_0005.html)
  uses RAS events, ECC evidence, repeated same-chip failures, and Ascend DMI
  stress results to distinguish possible hardware faults from software faults.
