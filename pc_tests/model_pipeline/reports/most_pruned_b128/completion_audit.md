# Completion audit: 19 October 2026

Both authorized batch128 comparisons are complete; no further experiments or
other batch sizes were launched.

- Random controls:495 unique full-test sweeps;24 fully paired settings and72
  summary rows. All120 fresh learned sweeps reproduced their original model
  selection counts. Uniform random mean accuracy over15 unique trials was
  36.2547%. Learned minus frequency-matched mean accuracy ranged from-0.336 to
  +0.210 percentage points across24 settings: no consistent accuracy advantage.
- Most-pruned controls:110 unique sweeps:21 p90 models x5 repeats and5 fresh
  R101 p0 FP32 reference sweeps. Script/static report hashes match provenance.
  All21 models have complete accuracy, latency and NVML GPU-energy results.
- p90 accuracy ranged2.17%-15.49%. Best observed was MobileNetV3-Small p90 FP32
  at15.49%, versus85.80% reference. None met any0.5/1/1.5/2pp allowance. All24
  pre-test validation-selection entries among p90 controls were null; this is
  retained rather than substituting test-selected models or a fallback.
- Current most-pruned latency/energy savings are against the fresh R101 p0
  FP32 reference, not the routing matrix's eligible-static alternative. The very
  large savings are accompanied by severe accuracy failures, not useful wins.

## Measurement limitations

During the random study NVIDIA telemetry explicitly reported software thermal
slowdown and software power capping (19October, about16:05Rome), despite core
temperature74C. The R101 FP32 reference sweep changed from about15.4s near the
start to58s later. Retain all data, but treat latency/energy savings as descriptive
under nonstationary performance, not definitive controlled routing speedups.
Alternating order cannot fully remove this confound. Most-pruned measurements
followed later, so do not directly interpret cross-study timing differences as
causal benefits of a policy.

Actual fresh runtime predictions differed from cached static predictions1498
times across4.95million image evaluations in the random study, and315 times
across1.1million evaluations in the most-pruned study. The input boundaries
differ (RGB128 GPU-normalization versus original normalized inputs), and
precision/batch execution can also differ; the exact cause was not isolated.
Measured accuracy uses actual runtime predictions, not cached accuracy. These
counts include repeated images/repeats and are not independent observations.

All failures, negative savings, raw checkpoints and hashes remain preserved.
No GitHub push, Jetson access, power-mode changes or new experiment is authorized.
