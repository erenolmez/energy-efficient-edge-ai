# Dedicated batch-one engine/profile ablation

User authorized investigation of runtime/training/measurement constraints on
28October. First isolate original min/opt/max1/32/128 vs dedicated1/1/1 TensorRT
profiles on a small fit-only selected6candidate pool, not213rebuilds.

Worker `experiments/profile_b1_engines.py`, launcher `scripts/start_b1_profiles.ps1`.
Atomic `study.json` here; engines, logs and exactlaunch_identity.json under
`artifacts/b1_engine_profiles_20261028/`. Preserve every originalgraph/engine/
weight/report and cancelledbaseline. Stopreceipt remains authoritative.

Selection frozen from original seed42fit positions only: FP32reference,cheapest
1ppeligible energy/latency candidates,then greedy complementary correctness
coverage with measuredenergy tie-break,until6unique. No testdata/labels opened.
Build with samegraphs/precision/workspace/TF32policy; only optimizationprofile
changes. INT8 uses originalpreparedQDQgraph, requires actualintegercomputepaths.

Per candidate:100fixed calibrationimages produce old/new finite logits, raw
maxdifference andargmaxdisagreements. Different tactics/precision can change
predictions: discrepancies are reported, not called exactparity or discarded.
Refresh validationcorrectness before downstreampolicycalibration fornewengines.
Five paired alternating original/new warmwindows,each>=2s, pinnedRGB128 toGPU
normalization/classifier/hostargmax; no router. GrossNVMLGPUenergy only,notCPU
orwholePC. Setup/load+contextenergy/time recorded separately from100probes and
warmtiming. Old/newsetup is repeated underindependent memorycleanup; record
GPUtemperature/power viaNVMLmetrics,neverchangeclock/powermodes.

Then hold the SAME6enginepool simultaneously fororiginal/new profiles tocompare
observeddevicefree/usedmemory,1GiBreserve, noeviction. Notabsolutepeakmeasurement.
Checkpoint eachbuild/profile/pool stage. Exactguard60s,2consecutive>=85C latch
nextsavedcheckpoint/gracefulshutdown;CPUtemperatureunavailable. No blindrestart.

Noheldoutaccuracyclaim orroutergain is established by thispreflight. The next
separate measuredruntime study includescompactCPUfamilies,currentCNNcontrol,
residency/coldwarm/cacheaware/hostbytes/cleanup/graphs/sync ablations withfrozen
validationpolicies. No E:,Jetson,smartplug,powermodes,HPCsubmission orGitHubpush.
