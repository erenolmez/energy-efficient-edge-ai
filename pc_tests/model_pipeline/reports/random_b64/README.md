# Batch64 random controls (user-authorized19October2026)

Same comparison as batch128, isolated outputs. No rerun/overwrite of prior
random_b128, most-pruned, scalar, or classifier/static studies. No training.

24 frozen correctness settings:3seeds42/45/48 x2objectives x4allowances
0.5/1/1.5/2pp at batch64. Same original seed+64 test shuffle,156full batches
and one real16-image tail. Five trials; fresh R101FP32 reference, original
validation-selected eligible static, fresh learned correctness router and
frequency-matched random controls percase. Uniform random all213 shared
by batch-order seed,5trials perseed, not repeated for eachobjective/allowance.
Total495 full10000-image sweeps. All213 candidates selectable, residentcapacity2.

Matched random preserves original learned image/model counts and tail model,
permuting model assignments over full batches without image/label information.
This is a post-hoc conditional randomization, not an independently learned
policy. Verify fresh learned selectioncounts reproduce source; disclose any
differences. Random paths have no neuralrouter; random schedule generation is
setup, dispatch lookup remains timed. PinnedRGB128 transfer/normalization,
enginehash/load/eviction/inference/sync included. No cached learned decisions.
Cold cache after3warmup batches; order alternatesbytrial. NVMLGPUenergy only.
Accuracy usesactual freshpredictions; recordcachedpredictiondisagreements.
PriorGPUthrottling/temporaldriftlimitsstronglatency/energyclaims.

Worker run_random_routes_b64.py uses tested bounded thermalJSONreadretry,
failsclosedonpersistenterrors andneverignoreslatchedthermalstates. Exact fresh
guard must trackthisscript, currentPIDs/executables/creationtimestamps, not old
identities. Guard reportpath is thisfolder's random_routes_b64.json. Thermal
state/log artifacts/pc_routers/thermal_guard.json/.log; samples60s, two
consecutive>=85C latch stopatnextsavedblock andpreviouslyauthorizedgraceful
shutdown. Neverrestartiflatched. CPUtemperatureunavailable.

Logs/identities: artifacts/random_b64/run.stdout.log,run.stderr.log,
guard.stderr.log,launch_identity.json. Checkbrieflyevery15minutes, update
completed/495,currentGPUtemperature, processstate; no continuouspolling.
Completion: summarize_random_routes_b64.py generates comparison table and
summaryJSON. Audit495unique blocks,24pairedcases,72summaryrows,all5trials,
provenance/protectedbatch128hashes,frequencycounts,discrepancies,accuracyfailures
andnegative savings. Notifyuser anddeletethismonitor. No E:,Jetson,smartplug,
powermodechanges,newbatches,oldexperimentsorGitHubpushauthorized.
