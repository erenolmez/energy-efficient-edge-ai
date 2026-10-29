# Batch-one correctness routing and fit-energy payback

User authorized 28 October: reuse existing trained weights; batch1 export/parity,
validation-only calibration, timing pilot, full learned/static comparison, then
plots and energy break-even using existing larger batches. No retraining or
rerun of the original213 static targets, random controls or training-energy run.

Worker `experiments/run_correctness_b1.py`; launcher
`scripts/start_correctness_b1.ps1`. Atomic state `study.json` in this folder.
Exact executable/script/creation identities and logs `run.stdout.log`,
`run.stderr.log`, `guard.stdout.log`, `guard.stderr.log` in
`artifacts/correctness_b1_20261028/`.

Stages:
1. Original seed42/45/48 weights, existing3000/2000 fit/calibration positions.
   Actual batch1 TensorRT engine,100random-input and100real-validation parity
   probes;213 sigmoid correctness probabilities. Calibration uses all2000
   reserved positions, never test labels. Existing b1 measured static costs and
   validation-only eligible static choices. All policies frozen before test opens.
2. Three256-image pilots using predefined latency/1pp policies, one perseed.
   Runtime extrapolation only; no test-based selection/tuning/protocol change.
3. Full10000-image test sweeps,3seeds x2objectives x4allowances(.5/1/1.5/2pp)
   x5trials x3roles(referenceR101FP32,eligible-static,learned)=360saved fullblocks.
   Preparation/pilot contribute6additionalcomplete stage rows. Report /360 full
   separately. Role order alternates with trial; same seed+1 imageorder percase.

Fresh GPUtransfer/normalization,TRT router,hostselection,capacity2 cache
hash/load/eviction,selectedinference/sync/hostpredictions. No cachedchoices;
all213 logical choices. Threewarmup decisions,then independent coldcache per
sweep. Loader indexing/pinning and decode/resize outside timers, as historical
boundary. New exact10000-image energy windows (no minimum-duration extra
sweeps),grossNVMLGPUonly;idle/display included,CPU/restPC excluded. Preserve
negative savings and accuracyfailures. Thermalguard every60s,2consecutive>=85C
latches nextsavedblock stop plus previouslyauthorizedgracefulshutdown. Worker
failsclosed;do notblindlyrestart. CPUtemperatureunavailable.

Analysis `analysis/summarize_correctness_b1.py`: CPUonly audit,24case summary,
combined144settings acrossb1/8/16/32/64/128,explicit eligible-static andFP32
comparators,perseed trainingjoules divided by positive savedJ/image. No valid
payback ifaccuracyfails;retain mathematicalcrossings separately. Fit-energy
excludes supervision/calibration/export and is newlymeasured,not historical
energy reconstruction. Crosssession GPUthrottling/nonstationarity caveat.
Never sum3trainingseeds as costof1deployedrouter. Plot remainingmeasurement
variability;no false uncertaintyclaim from only5repeats or3trainingseeds.

The PC CUDA environment lacks matplotlib. Plotting will use an isolated CPU
plot dependency directory under artifacts; do not change the inference/training
environment. Uniform random is deferred, not silently added to this protocol.
No E:,Jetson,smartplug,powermodes,HPCsubmission orGitHubpush.
