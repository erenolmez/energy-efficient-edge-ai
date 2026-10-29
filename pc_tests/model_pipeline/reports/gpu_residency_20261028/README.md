# Separate GPU residency probe, 28 October 2026

User stopped the multi-day batch1 TinyCNN study and authorized this standalone
capacity test. Preserve8fullsaved sweeps and6prep/pilot rows; stopped study and
original artifacts are never resumed/overwritten. Lifecycle receipt in
`reports/correctness_b1_20261028/stop_receipt.json` overrides historical running
field in the immutable originalstudy.json.

Worker `experiments/probe_gpu_residency.py`; launcher
`scripts/start_gpu_residency.ps1`. Atomic `study.json` here; exact executable,
script andcreation identities/logs in `artifacts/gpu_residency_20261028/`.

Keep seed42 tinyrouter resident. Progressively retain real original classifier
TensorRT engines AND executioncontexts/streams/batch1input-outputbuffers.
Verify trusted hashes, deserialize, execute finite synthetic(1,3,128,128)inputs.
No testlabels or accuracy claims; no training. Three frozen loadingorders:
smallestenginefilefirst,largestfirst,architecture-round-robin. No eviction during
eachpass; after loading verify ALL retained engines still execute with numerical
parity to their initial syntheticoutputs. Release all between passes.

Stop before crossing1GiBfreeGPUreserve using conservative filebytes+observedextra
allocation estimate (minimum64MiBextra). Roll back a candidate if its actual
postexecutionmemory violatesreserve. Do not intentionally forceOOM; unexpected
allocationfailures are preserved/not blindlyrestarted. The count is demonstrated
safe residency UNDERTHISORDER/RESERVE, not an absolutehardwaremaximum. Smallest
first may keep moremodels but does not establish a useful accuracy/cost pool.
Enginefiles total~3.27GiB is NOT GPUmemoryusage:contexts/buffers/driver addcost.

Log actual driverfree/usedmemory, observedincrement,load/contexttime andfirst
executiontime perengine; report peakobservedusage not unobservedhighwater.
This is batch1capacity, not largerbatchcapacity or a throughput/energyspeedup
experiment. GPU6GB includesdisplay/otherconsumers; freecapacity canchange.
Thermalguard60s,2consecutive>=85Clatchnextsavedrow/gracefulshutdown; no restart
afterlatch. CPUtemperatureunavailable. Activeguard expectedscript must be
probe_gpu_residency.py and exactidentities/telemetry<=130s.

Aftercomplete audit3ordercounts,reserve,enginehashes,allresidentparity andsource/
protectedreporthashes. Save smallreadablesummary; use results to choose safe
capacities for the separate authorizedcompactrouter study, not testlabeltuning.
30minmonitor remains active for residency/compactwork, not cancelledbaseline.
No E:,Jetson,smartplug,powermodechanges,HPCsubmission orGitHubpush.
