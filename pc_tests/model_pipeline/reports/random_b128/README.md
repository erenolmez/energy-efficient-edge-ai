# Batch-128 random controls (19 October 2026)

User-authorized comparison only; do not resume scalar studies, old experiment
automations, Jetson work, or other batch sizes. Original reports remain intact.

## Frozen comparison

- Original 24 correctness settings: seeds 42/45/48, latency/energy objectives,
  allowances 0.5/1/1.5/2 pp. Same seed+128 test shuffle, 78 full batches and one
  actual 16-image tail. All 213 candidates logically available, resident limit2.
- Uniform random: equal probability over all213, independent of image/labels.
  One set of five trials per batch-order seed; reused across objectives/budgets,
  not independently measured repeatedly for the same random policy.
- Frequency matched: source learned model counts permuted over full batches;
  retain tail model and exact image/batch counts. This is a post-hoc conditional
  randomization control, not a separately trained or validation-selected policy.
- Source counts are frozen from the original experiment, not chosen using new
  test outcomes. Fresh learned inference is remeasured and its counts checked.
  Report differences if counts change; do not silently redefine matching.
- Five measured trials per setting. Fresh reference, eligible static, learned,
  matched-random sweeps alternate order by trial. Uniform is shared by seed.
  495 full-test measurement blocks total. Each completed block is atomically
  checkpointed and resumable under the unchanged protocol/source hashes.
- No neural router in random paths. Random schedule generation/input pinning
  are setup; schedule lookup and all GPU preprocessing, hash/load/eviction,
  inference and synchronization are measured. Each block starts a cold capacity2
  classifier cache after three warmup batches. No cached learned decisions.
- NVML counter energy is GPU-only, not whole PC. Fresh RGB128 inference may
  differ slightly from saved normalized-input predictions; count disagreements
  and use actual runtime predictions for measured accuracy, not cached values.

## Files

- `cached_accuracy.json` / `cached_accuracy_table.md`: ten prespecified cached
  accuracy randomizations per case; no latency claims.
- `random_routes_b128.json`: actual measurement checkpoints and provenance.
- `random_comparison_table.md` / `random_comparison_summary.json`: measured
  comparison, clearly partial until all expected blocks finish.
- `../../artifacts/random_b128/`: logs and verified launch identities.

## Thermal safety and completion

`watch_pc_thermals.ps1` must track exact current run_random_routes.py executable,
script, PID/creation identities, with this report as the saved-block boundary.
Sample every60 seconds. Two consecutive readings at or above85C latch a stop;
save the next full sweep and allow the guard's previously authorized graceful
shutdown. Never restart a latched study. CPU temperature is not available.
Do not infer active status from old PIDs. No duplicate GPU workers.

After all495 blocks, run `summarize_random_routes.py`, audit counts/hashes,
actual random/frequency execution, prediction discrepancies and accuracy
failures, and notify the user. No GitHub push requested.
