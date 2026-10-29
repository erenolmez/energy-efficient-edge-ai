# INT8 batch-one diagnostic

Completed three matched passes on the same100 validation images. No test data
opened and no engines rebuilt or changed. Source and dedicated-engine hashes,
three unique completed rows and the no-test-data flag were checked after exit.

| Comparison | Changed predictions /100 | Maximum absolute logit difference |
|---|---:|---:|
| Original engine batch1 vs original batch32 |0|0|
| Original batch1 vs dedicated batch1 |40|5.9375|
| Original batch32 vs dedicated batch1 |40|5.9375|

The original engine's batch shape does not explain this discrepancy on the
probes. This does not identify the root cause inside the dedicated build, nor
establish equivalence or accuracy-preserving speedup. Dedicated ResNet34INT8
remains quarantined; fresh validation outcomes are being collected separately.
Original engines remain available for isolated runtime/switching comparisons.
