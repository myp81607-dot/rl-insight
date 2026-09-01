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

## Diagnostic vocabulary

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
  compute-heavy target slowdown with stable sequence length is only indirect
  support.
- `AI Vector overload`: contention concentrated in Vector compute. Direct
  AIV/Vector utilization or profiler evidence distinguishes it best. The
  current catalog alone may not reliably separate it from AI Core overload.

Evidence that weakens Compute: prompt, response, or global sequence length grows
with the latency; vLLM waiting or transfer backlog dominates without coherent
compute evidence; the event is a brief stall followed by rapid recovery.

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

Prefer Network when communication-heavy timing rises while throughput or MFU
falls and sequence-length features remain stable. Transfer-queue backlog is
propagation evidence, not proof of a parameter-plane fault. With the current
catalog, the first three sustained restrictions are usually observationally
equivalent; rank them as possible causes without pretending to locate the
restriction. Abrupt stall and rapid recovery weakly favor transient interruption.

Evidence that weakens Network: only rollout generation or vLLM metrics degrade;
data length grows coherently; or no communication-related timing/throughput
effect is present. Do not claim observed packet loss, rate limitation, or link
down without direct counters or logs.

### Host CPU

- `Host CPU core offline`: reduced scheduler-visible CPU capacity consistent
  with one or more host cores becoming unavailable.
- `Host CPU overload`: runnable work saturates available host CPU capacity and
  delays input, orchestration, serialization, or request handling.
- `Host CPU frequency throttling`: broad host-side work slows because effective
  CPU frequency is reduced.

Prefer Host CPU when CPU memory/resource evidence and host-sensitive latency or
queueing move together while NPU-oriented throughput effects appear secondary.
Direct CPU online-state, utilization/pressure, and frequency counters are needed
to distinguish the three causes reliably. The current catalog does not justify
claiming that a CPU was actually offlined, saturated, or frequency-limited.

Evidence that weakens Host CPU: isolated NPU compute evidence, isolated network
communication effects, or sequence-length growth fully explains the event.

### HBM

- `HBM congestion`: contention for NPU HBM capacity or memory bandwidth that
  causes compute and rollout work to wait. This label does not imply continuous
  saturation or defective memory.

Prefer HBM when NPU memory allocation/resource evidence, latency, and throughput
degradation are temporally coherent, especially when the severity varies over
time while sequence length remains stable. Allocated or reserved memory alone
does not measure bandwidth congestion. Direct HBM bandwidth/pressure evidence
is required for a high-confidence subtype claim.

Evidence that weakens HBM: stable memory evidence, a purely communication-shaped
stall, or a data-length increase that explains the memory and latency changes.

## Synthesis rules

1. Match the exact target event and phase before diagnosing. Confirmed and
   closed transitions receive separate reports.
2. Read all metrics in the grouped Top-25. Use direction, labels, temporal
   coherence, metric meaning, global rank, and contradictions; do not count
   category members as votes.
3. Treat data-characteristic changes as workload confounders before attributing
   latency to infrastructure. Treat training/rollout quality as downstream
   effects unless their semantics directly support a cause.
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
