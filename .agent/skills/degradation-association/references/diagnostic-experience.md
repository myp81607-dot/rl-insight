# Diagnostic experience

These diagnostic priors help interpret observed association evidence. They do
not assert direct observation of signals absent from that evidence. Do not use
this reference to provide operational procedures, commands, configuration
changes, parameter values, or steps for reproducing faults.

The evidence categories (`latency`, `training_quality`, `rollout_quality`,
`data_characteristics`, `hardware_resources`, `vllm_engine`, and
`transfer_queue`) organize metrics; they are not fault domains. Never select a
domain by category size alone. `association_percent` is temporal association
strength, not fault probability or causal contribution. Time-trend metrics such
as `training_epoch` and `tq_controller_uptime_seconds` require corroboration.

Use only the final `association_percent`, metric name, category, rank, target,
and event phase when applying these priors. Do not inspect raw samples, calculate
new statistics, compare before/after values, or infer whether a metric increased
or decreased. Several semantically consistent high-scoring metrics are useful
joint evidence; a category containing many weakly scored metrics is not.

## Diagnostic vocabulary

### Workload/data condition

- `Sequence-length anomaly`: consider this when several prompt-length,
  response-length, global-sequence-length, or total-token metrics are both
  high-ranking and high-scoring within `data_characteristics`. This is a
  workload/data explanation rather than an infrastructure fault domain. Do not
  claim that lengths increased, decreased, or changed by a particular amount;
  the association result alone does not establish direction or magnitude.

Strong sequence-length association weakens an infrastructure-only explanation.
One isolated length metric, or many low-scoring length metrics, is insufficient.

### Compute

- `NPU frequency throttling`: reduced active compute capacity consistent with a
  lower operating frequency. Prefer it for a relatively broad, sustained
  slowdown without evidence of a disappeared device or rank. Idle low frequency
  is normal and frequency must not be claimed unless directly observed.
- `NPU core offline`: the experiment label for abrupt, severe, persistent loss
  of effective NPU compute capacity. It may represent localized capacity being
  unavailable; do not translate this label into a claim of physical core
  disappearance, ECC/RAS failure, or device removal without a direct signal.
- `AI Core overload`: contention concentrated in Cube/matrix compute. Direct
  AIC/Cube utilization or profiler evidence distinguishes it best; otherwise a
  compute-sensitive target with strong MFU/throughput association and weak
  length/sequence association is only indirect support.
- `AI Vector overload`: contention concentrated in Vector compute. Direct
  AIV/Vector utilization or profiler evidence distinguishes it best. The
  current catalog alone may not reliably separate it from AI Core overload.

Evidence that weakens Compute: high-scoring length/sequence metrics, or stronger
vLLM/transfer-queue association without comparably strong compute-related
metrics.

### Network

- `Parameter-plane NIC bandwidth limitation`: sustained communication-capacity
  limitation associated with an endpoint network path.
- `Parameter-plane switch-port egress bandwidth limitation`: sustained
  communication-capacity limitation associated with an egress path.
- `Parameter-plane network congestion`: sustained or bursty contention within
  the communication path.
- `Transient parameter-plane link interruption`: abrupt communication stall,
  often followed by recovery or a closed event, consistent with a temporary
  link interruption.

Prefer Network when communication-sensitive timing and transfer-queue metrics
receive strong association scores while length/sequence metrics do not. Strong
transfer-queue association is propagation evidence, not proof of a
parameter-plane fault. With the current catalog, the first three sustained
restrictions are usually observationally equivalent; rank them as possible
causes without pretending to locate the restriction.

Evidence that weakens Network: association scores are concentrated in rollout,
vLLM, or length/sequence metrics and communication-related metrics rank weakly.
Do not claim observed packet loss, rate limitation, or link down without direct
counters or logs.

### Host CPU

- `Host CPU core offline`: reduced scheduler-visible CPU capacity consistent
  with one or more host cores becoming unavailable.
- `Host CPU overload`: runnable work saturates available host CPU capacity and
  delays input, orchestration, serialization, or request handling.
- `Host CPU frequency throttling`: broad host-side work slows because effective
  CPU frequency is reduced.

Prefer Host CPU when CPU resource and host-sensitive latency/queueing metrics
receive strong association scores while NPU-oriented evidence ranks lower.
Direct CPU online-state, utilization/pressure, and frequency counters are needed
to distinguish the three causes reliably. The current catalog does not justify
claiming that a CPU was actually offlined, saturated, or frequency-limited.

Evidence that weakens Host CPU: stronger NPU, network, or sequence-length
association without comparably strong host-sensitive metrics.

### HBM

- `HBM congestion`: contention for NPU HBM capacity or memory bandwidth that
  causes compute and rollout work to wait. This label does not imply continuous
  saturation or defective memory.

Prefer HBM when NPU memory resource, latency, and throughput metrics all receive
strong association scores while length/sequence metrics do not. Allocated or
reserved memory alone does not measure bandwidth congestion. Direct HBM
bandwidth/pressure evidence is required for a high-confidence subtype claim.

Evidence that weakens HBM: memory metrics rank weakly while communication or
length/sequence metrics dominate the association scores.

## Synthesis rules

1. Match the exact target event and phase before diagnosing. Confirmed and
   closed transitions receive separate reports.
2. Read all metrics in the grouped Top-25. Use metric meaning, category, global
   rank, and final association score only. Do not use raw values, direction,
   labels, component scores, or independently calculated trends.
3. Treat a coherent cluster of high-scoring length/sequence metrics as evidence
   for `Sequence-length anomaly` before attributing latency only to
   infrastructure. Treat training/rollout quality as downstream evidence unless
   its metric semantics directly support a cause.
4. Rank exactly two distinct fault domains and three to five specific causes;
   cause 1 is primary. `Low` confidence is allowed, but every item needs an
   evidence or uncertainty basis and must not contradict observed evidence. If
   the selected phase has no association entries, report insufficient root-cause
   evidence instead of fabricating this ranking.
5. Keep the reasoning concise and professional. Do not invent profiler, network,
   frequency, CPU, HBM, ECC/RAS, or device-health observations.

## Technical sources

- [Ascend AI Core architecture](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0015.html)
  describes Cube, Vector, and scalar compute units.
- [Ascend ArithmeticUtilization fields](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0093.html)
  distinguish AI Cube Core and AI Vector Core execution evidence.
- [Ascend PyTorch Profiler MemoryAccess](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0033.html)
  documents memory-access and bandwidth analysis.
- [HCCL alpha-beta model](https://www.hiascend.com/document/detail/en/canncommercial/850/commlib/hcclug/hcclug_000115.html)
  relates communication time to latency and per-byte transfer cost.
- [Linux CPU hotplug](https://docs.kernel.org/core-api/cpu_hotplug.html),
  [CPUFreq](https://docs.kernel.org/admin-guide/pm/cpufreq.html), and
  [Pressure Stall Information](https://docs.kernel.org/accounting/psi.html)
  define direct host CPU availability, frequency, and contention signals.
