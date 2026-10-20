# Most-pruned static comparison (queued by user19October2026)

After the random study finishes and is audited, run highest available pruning
for every architecture having pruning variants: p90 for seven architectures,
FP32/FP16/INT8, 21configurations. ShuffleNet has no pruned variant and is excluded.
Five complete10000-image sweeps per model plus five fresh R101 p0 FP32 reference
sweeps =110 checkpoints. Batch128 with real tail16, seed42+128 ordering, no router.

Timing/energy boundary matches the random study: pinned RGB128 transfer and
normalization, fixed model lookup, cold capacity2 engine hash/load, inference,
synchronization/host predictions. Three warmup batches before clearing cache.
GPU-only NVML energy, not whole PC. Reverse model order on alternate trials.
Do not infer MAC/parameter reduction or fastest execution from the p90 label.

Validation-only selection among21 is frozen from original2000 calibration
positions for seeds42/45/48, objective-specific prior costs and allowances
0.5/1/1.5/2pp. Null means no p90 model qualifies; never use test labels to fill it.
Measure all21 regardless of eligibility and preserve all accuracy failures.
Reported savings are against freshly measured R101 p0 FP32, explicitly different
from routing-table eligible-static savings. Separate study dates/clock drift mean
cross-study latency comparisons are descriptive, not controlled proof of wins.

Start via pc_tests/model_pipeline/start_most_pruned.ps1 only after495/495 random
blocks and72 complete summary rows, old worker/guard exited, thermal not latched.
The helper verifies no active competing GPU experiments and launches hidden
run_most_pruned.py with a matching exact-identity60s thermal guard. Record process
identities in artifacts/most_pruned_b128/launch_identity.json, never reuse old PIDs.
Read run.stdout.log, run.stderr.log and guard.stderr.log there.

Thermal policy: two consecutive>=85C readings latch stop at next saved sweep and
request previously authorized graceful shutdown; never restart while latched.
CPU temperature unavailable. Notify progress every15minutes, no busy polling.
No Jetson, E:, smart plug, power-mode changes, older jobs or GitHub push authorized.

The runner writes most_pruned_routes.json atomically per sweep, then generates
most_pruned_table.md and most_pruned_summary.json. Rebuild summary without GPU
using run_most_pruned.py --summarize-only. Finish only at110/110 and21 summary rows.
