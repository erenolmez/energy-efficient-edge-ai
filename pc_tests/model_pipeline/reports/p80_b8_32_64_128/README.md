# Fixed p80 static study

User requested p80 instead of p90 on 19 October. Keep all p90 outputs unchanged.
Run only after batch64 random finishes, its summary is audited, worker exits and
its guard exits. Do not overlap measurements or restart a latched thermal state.

Launcher: `pc_tests/model_pipeline/start_p80_batches.ps1`.
Worker: `run_p80_batches.py`, exactly 21 p80 configurations: seven architectures
x FP32/FP16/INT8. All models run; none is selected using held-out labels.
Batch sizes 8,32,64,128, five full 10000-image repeats per model and five fresh
R101 p0 FP32 controls at each size: 440 atomic checkpoints,84 model/batch rows.
Validation-only candidate eligibility is frozen before runtime evaluation.
True tail16 at32/64/128. Seed42+batch ordering, odd-trial model order reversed.
Same pinned RGB128 transfer/normalization, cold capacity2 hash/load/inference,
host predictions and synchronization boundary as prior p90 controls. Setup and
pinning excluded. GPU-only NVML energy. Preserve every accuracy failure and
negative saving; costs are repeat medians, accuracy repeat mean. Compare savings
with fresh same-batch R101 FP32, not a mismatched seed/budget eligible comparator.
Compare accuracy to original 13 October rows and fixed reference; separate-time
latency/energy comparisons remain subject to GPU throttling/nonstationarity.

Report: `p80_routes.json`; summarize with `run_p80_batches.py --summarize-only`.
Logs and exact launcher/worker/guard identities: `artifacts/p80_b8_32_64_128/`.
Guard must expect `run_p80_batches.py`, verify exact executable/script/creation
identities and refresh telemetry every60s (fresh<=130s). Two consecutive>=85C
latch a stop at next saved checkpoint then previously authorized graceful
shutdown. Never restart waiting/latched/stopped/shutdown states. CPU temperature
unavailable. Preserve protected p90/random/static/test hashes and scientific
protocol; do not blindly resume failed workers or change the protocol.

User-requested progress cadence remains15minutes: brief check then end turn.
Completion audit:440unique sweeps,110eachbatch,21x5models+5references perbatch,
84summaryrows,allsource/input/protectedreporthashes,cachedpredictiondifferences,
allaccuracyfailuresandnegatives. Replace p90 table in20October report only with
audited actual p80 results, not cached static estimates or placeholders.
Explicit C: paths only. No E:,Jetson,smartplug,powermodechanges or GitHub push.
