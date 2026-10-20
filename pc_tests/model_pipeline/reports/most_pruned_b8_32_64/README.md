# Most-pruned static controls: batches8/32/64

Authorized19October2026 after completion of batch128. Preserve batch128 report
and random comparison unchanged. One sequential run_most_pruned_batches.py GPU
worker, expected330 full10000-image sweeps:21 p90models x3batchsizes x5repeats
plus15fresh R101p0FP32references. Sevenarchitectures xFP32/FP16/INT8. ShuffleNet
excluded because it has no pruning variants. No router, training or model builds.

Seed42+batch input shuffle; batch8 has1250full batches;32 has312full+tail16;
64 has156full+tail16. Input/RGB128 timing boundary, coldcapacity2, warmup3 and
NVMLGPUenergy match prior static comparison. Each block is atomically saved.
Accuracy is actual fresh runtime output; record differences from cached static
predictions. All failure/negative savings retained. Summary cost medians use
same-batch fresh R101FP32reference, NOT eligible-static routing comparators.
Observed prior throttling/nonstationarity limits cross-study timing claims.

Run .cuda-venv\Scripts\python.exe pc_tests\model_pipeline\run_most_pruned_batches.py
only with a fresh matching watch_pc_thermals.ps1 guard. ExpectedScript must be
run_most_pruned_batches.py; ReportRelativePath must point to this folder's
most_pruned_routes.json. Launch identities/logs are in
artifacts/most_pruned_b8_32_64/launch_identity.json,run.stdout.log,run.stderr.log,
guard.stderr.log. Verify current executable/script/creation identity, never old
PIDs. Guard every60s; two consecutive>=85C latch stop at next complete sweep
then previously authorized graceful shutdown. Never restart latched study.
CPUtemperature unavailable. No E:, Jetson, smartplug, powermodechanges or push.

Monitor briefly every15minutes and report completed/330, GPUtemperature and
processstate. No continuous polling. At330/330 statuscomplete, run
run_most_pruned_batches.py --summarize-only and audit110sweeps perbatch,
five repeats permodel/reference,63summaryrows, provenance hashes and preserved
batch128hash. Notify accuracy/latency/energy table paths and delete this monitor.

## Read-race repair and authorized continuation

On19October the first worker stopped at127 saved sweeps after reading the guard
JSON during its non-atomic rewrite. No thermal latch occurred. The user explicitly
requested continuation. A bounded20x25ms retry now handles transient empty/partial
JSON or read-sharing errors, never cached telemetry; persistent errors fail closed
and latched statuses still stop execution. Five CPU tests pass. Original source,
report and launch identities are backed up in artifacts/most_pruned_b8_32_64 with
before_readfix filenames. All127 previous rows and scientific protocol remain
unchanged; code_revisions records original/current source hashes and reason.
Current logs are resume.stdout.log,resume.stderr.log,resume.guard.stderr.log,
listed in launch_identity.json. Verify the latest identities, not original PIDs.
Completion source audit must accept only the explicitly recorded repair lineage,
while all static/test/protectedbatch128 hashes must remain identical.
