# Authorized compact batch-one routers and switching study

## Completed, 29 October

All authorized bounded investigations, saved-result audits, four final scientific
figures and23 CPU regression tests are complete. The final summary, indexed
audit receipts, fit-specific gated payback and completion receipt are under
`final_summary/` (`results.md`, `supervision_payback.json`, `completion_receipt.json`).
All four new PNGs were visually inspected. The cancelled multi-day baseline
remains cancelled. No additional experiments are queued. The monitor is being
removed and a non-forced Windows shutdown requested under the user's authority;
`final_summary/shutdown_receipt.json` records the request, not proof of power-off.

## Historical checkpoint, 29 October 03:18 UTC

The final supervision deployment measurement is COMPLETE:20/20 full sweeps,
400 chunks and6 export/parity preparations. `analysis/summarize_supervision_deployment.py`
passed the source/plan/protected/test/engine/weight hashes, saved predictions,
selection counts, accuracy and positive-energy checks. Results are in
`reports/supervision_deployment_20261029/{audited_summary.json,comparison.csv}`.
No additional GPU experiment is authorized or needed for this checkpoint.
Remaining before completion shutdown: fit-specific payback, final scientific
plots (including batch8 and supervision), consolidated task checklist/report,
visual inspection and CPU regression tests. Do not shutdown just because the
last measurements finished. Preserve all failures and negative savings.

## Current stage after explicit human resume, 28 October

The pause was explicitly cleared after all187 checkpoint hashes matched. The
cancelled multi-day baseline remains cancelled. Residency,12CPU fits and all
six engine-profile comparisons are complete and must not be rerun.

The additional INT8 diagnostic completed with3unique saved rows: original
batch1 and original batch32 give identical logits on100 fixed validation
images; dedicated batch1 differs on40 classes (max logit difference5.9375).
This rules out original-engine batch shape as the explanation on these probes,
but does not isolate the dedicated engine's numerical cause. INT8 stays
quarantined from deployment comparisons, not silently replaced or discarded.

COMPLETED/AUDITED: `experiments/refresh_compact_validation.py`, launcher
`scripts/start_compact_validation.ps1`. Exact identities/logs are in
`artifacts/compact_validation_20261028/launch_identity.json`; atomic progress is
`reports/compact_validation_20261028/study.json`. Guard expected_script must be
`refresh_compact_validation.py`. First check same-original-engine file,
verified-hostbytes, shared-buffer and CUDAgraph adapters on100 validation
probes for each of the6frozen models (24adapter rows, unsupported graphs explicit).
Then refresh5000validation predictions per dedicated engine in500image atomic
checkpoints (60chunks total), saving predictions/hashes for all3original splits.
This is validation preflight, NOT heldout accuracy or timing/energy savings.
The dedicated INT8 remains quarantined even after its validation is measured.
Do not duplicate/restart this worker or old diagnostic. After it and its guard
exit, audit outputs and proceed with original-engine samepool runtime timing,
switch profiling and sequential policy calibration; new engines require fresh
targets and explicit numerical acceptance, not old cached correctness.

COMPLETED/AUDITED STAGE: `experiments/profile_compact_runtime.py` launched through
`scripts/start_compact_runtime_profile.ps1`. Exact identities and logs are in
`artifacts/compact_runtime_profile_20261028`, report in
`reports/compact_runtime_profile_20261028/study.json`; guard expected_script is
`profile_compact_runtime.py`. Uses only ORIGINAL engines for a clean comparison:
36 file/verified-host load measurements,60 eager/graph paired2-second warm
windows,60 fresh-original-validation500image chunks plus1hoststore setup =157
checkpoints. Six fixed models, no test images. Warm windows include fresh GPU
normalization and finalhostclassID; cold load/setup/cleanup reported separately.
After completion audit measurements and final arrays, then use these measured
costs/fresh correctness for sequential calibration and compact-vs-CNN comparison.
Dedicated INT8 still quarantined; all4original-engine adapters passed parity.

COMPLETED/AUDITED: `experiments/measure_gpu_dispatch.py` via
`scripts/start_gpu_dispatch.ps1`. Report `reports/gpu_dispatch_20261028/study.json`,
exact identities/logs `artifacts/gpu_dispatch_20261028/launch_identity.json`.
Guard expected_script `measure_gpu_dispatch.py`. Fixed seed42 engineering
comparison, original6resident classifiers, original213-output CNN. Calibrate
latency/1pp on original2000cal positions using fresh-original correctness and
new measured costs. Compare full213probability CPUreadback/selection against
GPU6pool selection and scalarIDreadback, samehostlaunch/finalclass, fivepaired
alternating2second windows.100probe decision/finalclass parity required.
13checkpoints target. This is NOT a full213choice or heldout accuracy study.
Full compact router and cache-aware confirmations remain pending afterward.

Runtime profile audit retained14short setup-energy windows as unavailable
because NVML counter did not advance; NOT zero energy. All60warm windows had
finite positive GPUenergy. Setup latency remains usable; aggregate cold-energy
measurement is still pending. Warm classifier-only graphs saved latency about
1.9% to49.9% across6models; retain negative GPUenergy saving forR101FP32.

COMPLETED/AUDITED: `experiments/run_compact_resident_screen.py`, launcher
`scripts/start_compact_resident_screen.ps1`, report
`reports/compact_resident_screen_20261028/study.json`, exact identities/logs in
`artifacts/compact_resident_screen_20261028/launch_identity.json`. Guard must
expect `run_compact_resident_screen.py`. Preliminary latency/1pp screen only:
all3seeds x(fourCPUmethods+samepoolTinyCNN+freshR101reference+freshvalidation
eligible static)=21 full10000image sweeps,420 atomically saved500image chunks.
Three calibration rows plus resident setup precede tests. Plan/hashes frozen
before opening test arrays. Original6engines are all resident, eager execution;
independent CPUheads evaluate only6available models, whereas sharedtree/MLP/CNN
retain full213output compute. Subset parity unit test passed for all4families.
Fresh CPUfeatures/prediction/selection, GPUpreprocessing, classifier andhostclass
are timed; setup separately, no loads/evictions within warm sweep. GPU-only
energy cannot measure CPU energy or establish CPUtraining payback. This is NOT
the cancelled360sweep/capacity2/213choice matrix. No test-based model/policy tuning.
After completion audit21sweeps/allseedmethods,420chunks, immutableplan/protected
hashes,counts,accuracy,positiveenergy; retain negatives andaccuracy failures.
Further switching-aware/cachecapacity/graph confirmations anddata/training
ablations remain pending, not complete from this preliminary screen.

Resident screen audit: all21fullsweeps/420chunks and hashes verified.5/15router
cases pass1pp accuracy; no routed case bothpasses andimproveslatency+GPUenergy
againsteligible static inthispreliminaryscreen. TinyCNN42fits GPUenergy pays
back at2399images vsR101FP32 only; no validTinyCNNpaybackvseligible static.
Allseeds/failures retained. CSV/JSON/results andvisuallycheckedPNG/SVGfigures:
`reports/compact_resident_screen_20261028/{comparison.csv,audited_summary.json,
results.md,compact_comparison.png,tinycnn_fit_payback.png}`. One repeat only.

COMPLETED/AUDITED: `experiments/pilot_compact_switching.py`, launcher
`scripts/start_compact_switching_pilot.ps1`; reports andartifacts directories
`compact_switching_pilot_20261028`, exactlaunch_identity/logs inartifacts.
Guard expected_script `pilot_compact_switching.py`. Predeclared seed42
validation-only256image pilots: file/staticcapacity2,file/cache-awarecapacity2,
host/staticcapacity2,host/cache-awarecapacity2,host/no-forcedcleanupcapacity2,
allresident6eager,allresident6graphs. OriginalTinyCNN, original6classifiers.
Policies calibrated on2000originalcalpositions; new measuredmisscostincludes
observedcleanup. Sevenpilots plusonepolicyfreeze checkpoint. No testdata.
Profiledbytebudget and1GiBreserve checked. Projectedfullsweep time isnotactual
fulltestgain. Auditpilots before smallconfirmation, neverblindrerun failedjobs.

COMPLETED/AUDITED: `experiments/confirm_compact_switching.py`, launcher
`scripts/start_compact_switching_confirmation.ps1`. Reports/artifacts directories
`compact_switching_confirmation_20261028`; exactlaunch_identity/logs inartifacts,
guard expected_script `confirm_compact_switching.py`. Fixed smallconfirmation:
3seeds x3repeats x(5routedvariants+4matchedstaticcontrols)=81full10000image
sweeps,1620atomic500image chunks. Fivevariants: file/staticcapacity2,
host/static/no-forcedcleanupcapacity2,host/cache-awarecapacity2,
6residenteager,6residentgraphs. Controls: freshR101FP32/validationeligible each
eager ANDgraph, so graphroute is not credited against onlyunoptimizedcontrols.
Allpolicy/eligible choices frozenbeforetestaccess; original6engines/TinyCNN.
Host-aware recalibration explicitly uses measuredHOSTmiss+cleanupcost rather
thanfilemisscost. This is a predefinedmechanism shortlist, notchosenfromtest
scores. All3seeds retained. No compactCPUrefits or cancelledmatrix resumption.
Audit81sweeps/1620chunks/plan/protected/test/sourcehashes,predictions/counts,
actualloads/evictions/GPUenergy; summarizeall5variantsx3seeds withmedian/ranges
across3repeats vsbothmode-matchedstaticbaselines, retainaccuracyfails/negatives.

RECOVERY28October: Windowsreader-lock WinError5 stoppedconfirmation at41full
sweeps plus5partialchunks. Originalsource/report/plan/identity preserved under
`artifacts/compact_switching_confirmation_20261028/recovery_checkpoint41`.
FiveCPUtests passed for newisolated`runtime/atomic_checkpoint.py`: bounded
40x25msreplacementretry, persistentfailurepreservesoldcheckpoint (neverforce
delete). Originalglobalwriter unchanged. `analysis/prepare_switching_resume.py`
verified41NPZs/predictions/chunks/counts/source/plan/protected/testhashes and
recorded`resume_receipt.json`/`code_revisions`. All41completedrows remainbyte-
equivalent aftercanonicalJSONserialization;5partialchunks archivedunder
`interrupted_attempts`, notmixedintonewfullsweepmetrics. ResumeusesUNCHANGED
frozenpolicies, skips41completedsweeps, restartsoneincompletefullsweeponly.
Latestlauncher`resume_compact_switching_confirmation.ps1`; workerargument
`--resume`. CURRENTlogsare`resume1/run.stdout.log`,`resume1/run.stderr.log`,
`resume1/guard.stderr.log`; currentexactidentitysameprimarylaunch_identity.json
andresume1copy. Oldrun.stderr.log ishistorical, notanewfailure.
Latesthumaninstruction: continueauthorizedwork, GPUthermalmonitoring; graceful
PCshutdown ifthermalstoptriggers OR ALL authorizedtasks/audits/plotscomplete.
Do notshutdownmerelybecause81sweepstagefinishes. Neverforceunsavedapps.

Confirmation nowCOMPLETE/AUDITED81/81,1620fullchunks,3calrows; source-repair
lineageand41originalrows checked.7/15routercasespassaccuracy. Two recordboth
positivesavingsvsmode-matchedeligible-static, butallhost-aware policies collapse
to constantR101FP16 (10000/10000images); their timinggain isNOTdynamicselection
evidence. Eligible-static timings variedsubstantially; no general speedupclaim.
Seed42residentgraphrecords9.07%latency/4.20%GPUenergysavingsandaccuracy pass,
notestpolicyselection. FullCSV/JSON/Markdown andvisuallycheckedtwoPNG/SVGplots
in`reports/compact_switching_confirmation_20261028`. Allnegative/failedcaseskept.

COMPLETED/AUDITED: `experiments/screen_router_supervision.py`, launcher
`scripts/start_router_supervision_screen.ps1`. Reports/artifacts directories
`router_supervision_screen_20261029`, exactlaunch_identity/logs inartifacts,
guard expected_script `screen_router_supervision.py`. Seed42validation-only
predeclared5fits:1000/2000/3000nestedclass-balanced originalfit positions,
20epochs; plus separate2400fit/600internal-tune unweightedandbalanced60epoch
fits,20/60checkpoints selectedonlyusinginternal600poolBrier. Originalreserved
2000calpositions untouchedbytraining;500/1000/2000balancedpolicycal subsets,
remainder heldoutVALIDATION whenavailable.180epochcheckpointrecords+5fitrows.
No testdata, no classifierretraining, no originalweightsreplacement. Original213
cachedtargets except6currentoriginal-engine columns refreshed consistently.
BalancedBCE fit-onlyposweights clipped[.1,10], predefinedlogweight subtraction
atprediction. Actual5GPUfits,180losses,7snapshotweights/calNPZ/provenance mustbe
audited; latestepochfiles overwrittenbydesign, snapshot20/60filespersist.
TrainingGPUenergyincludesevalinternal-tune/checkpointwrites, excludespolicycal/
export/test/CPU. No traineddeploymentclaimuntilactualexport/parity/runtime.
ThreeCPUdesign tests passed. NoCPUplots/fits overlap measuredGPUtraining.

Supervision auditpassed5fits/180epochrecords/15policycalibrationgroups.
PoolBrier1000/2000/3000 fits:.157959/.157512/.157081. Internal600tune chose
60epochcheckpoint forbothunweighted andbalanced models, poolBrier.157040/
.157011 onreservedcal. Smallvalidation-onlydifferences, nodeploymentclaim.
Seven20/60snapshots+5latestepochfiles/hashchecked; intermediateepochfiles are
overwrittenbydesign. Newweights notexported/runtime-validatedyet. No testdata.

COMPLETED/AUDITED: `experiments/compare_batch8_grouping.py`, launcher
`scripts/start_batch8_grouping.ps1`. Reports/artifacts directories
`batch8_grouping_20261029`, exactlaunch_identity/logs inartifacts; guard must
expect`compare_batch8_grouping.py`. Threeoriginalseeds, originalweights/six
originalresidentclassifiers. Newperimage-output batch8routerexports(8,213),
randomparity+realprobability-meanparity againstoriginalaggregatebatch8router
beforetest. Freshclassifier2000caloutcomes,actualgroupedcal1ppgate;
validation-failedpolicy fallsbackto reference BEFORE testaccess. Frozenplan.
Modes batch_choice/per_image/reference/eligible_static x3seedsx3repeats=
36full10000image sweeps,900atomic400imagechunks. Perimage modefreshCPUchoices
plusGPUgather/infergroups/scatter/reassembly allcharged; samefinal8hostclasses.
No test-tuning, noimagecopyback, nochargingcachedchoices. NewexportsNOTnewfits.
Audit36sweeps/900chunks/3prepare rows,plan/test/protected/weights/enginehashes,
counts/predictions,actualgroupinvocations andfiniteGPUenergy; compareactual
accuracy/cost andlatencyperimage/perbatch; keepnegatives/failures/fallbacks.

Batch8auditpassed36sweeps/900chunks/3export+paritypreparations. Perimagegrouping
improvedaccuracyoveronebatchchoice,butrequires2570/2659/2801classifiercalls
vs1250per10000images and2.77–3.28ms/image vs0.77–0.97onebatchchoice.
Onlyseed42perimagepasses1pp;all3batchchoiceandother2perimagefailaccuracy.
No efficientdynamicroutingclaim; allcontrols/negatives retained inCSVJSONMD.

COMPLETED/AUDITED: `experiments/profile_dispatch_controls.py`, launcher
`scripts/start_dispatch_controls.ps1`, reports/artifactsdirectories
`dispatch_controls_20261029`, exactlaunch_identity/logs inartifacts. Guard
expected_script`profile_dispatch_controls.py`. Validation-onlyengineering:
5paired512image sameR101FP16engine/context A/Brepeats(16imageinterleavedblocks),
staticvsactualCNNreadback/costcalculationthenforcedsameclassifier. Actual
FP32TensorRTselectorengine (int32typedoutput),512realchoiceparity vsFP64,
105neartiefixtures,3repeatsx3selectorpathswithfixed512inputs+freshselected
classifier. PreserveFP32choicemismatches, don'tsilentlyequatetoFP64. Six>=2s
repeatedload/context/cleanupenergywindows(filevshost) estimatesamortizedGPUJper
LOAD,notinferenceimage;neverbackfillold14unavailableindividualsetupreadings.
NVMLtemperature/clocks/power/pstate/throttlemaskendpointlogging,nochanges.
ReadonlyCPU/systempowersensoravailability check, nocounterenergyclaimfrom
instantaneouspower;nodriverinstallation/noexternaldeviceaccess.21rows target.
Auditallrows/actualfixedcounts/parity/energy/scope,sameclassifierABoutputs;
then trueGPUconditional2branchprototype remains pending. No completedreruns.

Dispatchcontrols21rows audited. PairedsameclassifierCNN/readbackextra.44–.75ms
inall5pairs, butstaticlatencyvaried4.10–7.12ms; no universal static timingclaim.
CPU64/TorchGPU64/TRTFP32 median4.352/4.368/4.390ms overlapranges; no reliable
GPUselectorbenefit. FP32changed0/512realchoices but15/105deliberatenearties.
Aggregatecoldfile/host load+cleanup median.439/.157s and6.584/2.508GPUJ/cycle,
notpurestartupJ orJ/inference. Old14unavailable setupreadingsremainunchanged.
BothcheckedWMIpowerprovidersunavailable; noverifiedCPU/systemenergymeter and
noCPU/fullPCenergy/paybackclaim. No driverinstall orpowerchanges.

COMPLETED/AUDITED: `experiments/probe_gpu_conditional.py`, launcher
`scripts/start_gpu_conditional.ps1`; reports/artifactsdirectories
`gpu_conditional_20261029`, exactlaunch_identity/logs inartifacts; guard must
expect`probe_gpu_conditional.py`. Boundedvalidation-onlyseed42engineering:
twoFP32classifiers R101/EffB0, sameoldCNNweights, standalonefixedb1rebuilds and
combinednativeONNXIf (notWhere). Costscoremedianbias fromcalibration deliberately
exercisesbothbranches, NOT anaccuracy-budgetdeploymentclaim. Nativeconditional
inputboundaries andALLbranchConv/MatMul lazydependency required. Actual512image
chosenbranchlogit/classparity, typedGPUbranchIDdiagnostics, IProfilertrace for
bothbranches. Onlyifruntimebranchtraceverified time3repeatsCPUdispatchvsGPUIf
vsboth-eagernegativecontrol, same512inputs/finalCPUclass; no profilerintiming.
Unsupportedparser/build/lazystructure/traceparitylimitations retainedhonestly;
do notfabricate fullyGPUrouting/speedup. No testdata/classifierretraining or
originalengineoverwrite. CPUONNXIflexicalcapture/outputsemantic unit testpassed.

Conditionalprototype15rowsaudited: nativeIf lazyclassifiervolumeboundaries,
runtimeprofileverifybothbranchesindividually,unselectedConv/Gemm0;512chosenbranch
logitparitymax1.79e-5,0class/CPUchoicechanges. GPUconditionalmedian4.123ms vs
CPUdispatch3.955ms (4.24%slower),both-eagernegativecontrol6.596ms. Genuine
application-levelconditionaldispatch, NOTgeneralperformanceimprovement.
No PythonIDreadback intiming; noassertionthatallTRTinternalsavoidhostsync.
Engineeringmedianbiasexercisesbothbranches;NOTvalidated1ppdeploymentpolicy.

CURRENT ACTIVE: `experiments/validate_supervision_deployment.py`, launcher
`scripts/start_supervision_deployment.ps1`. Reports/artifacts directories
`supervision_deployment_20261029`, exactlaunch_identity/logs inartifacts;guard
expected_script`validate_supervision_deployment.py`. Lastsubstantialmeasurement
stage: fivepredeclaredseed42supervisionfits actuallyexportedTRTbatch1,100random
and2000realvalidationprobabilityparity (balancedlogweightcorrectionpreserved).
FreshTRTcalibration500/1000/2000 beforetest =15cases, plusoriginalCNNcontrol2000,
freshR101reference and3cal-size-matchedeligible-staticcontrols=20fulltest
sweeps,400500imagechunks+6prepare rows. Onepasspreliminaryscreen,noseed/test
selection,noretraining,nocancelledmatrixresume. Same6residentoriginaleager
classifiers, freshGPUpreprocess/router/CPUselection/classifier/hostclass.
Frozenplan/protected-training/test/weights/enginehashes. Audit20sweeps/400chunks/
6paritypreparations, all16routedaccuracyfailures/negatives,fit-specificGPUenergy
paybackonlyifactualpositivesavings+accuracypass. GPU-onlyscope/fullpipeline
exclusions andseed42only/one-repeatlimitsmustremainclear. Thenconsolidate
allstudyresults/scientificplots/checklist,verifyallremainingauthorizeditems,
savecompletionreceipt,remove/pausemonitorandgracefulPCshutdownasuserrequested.

28 October: user authorized all four proposed compact router families and all
six switching-reduction avenues, with updates every30minutes. This is a NEW
study, not a mutation of the running October13-compatible batch1 baseline.

## Execution queue and current implementation status

1. Latest user instruction STOPPED the long `correctness_b1_20261028` baseline.
   Its8savedfullsweeps and6prep/pilotrows are retained; neverresume. The separate
   `gpu_residency_20261028` probe is now first. No overlapping GPU jobs or real CPU fits,
   memory profiles, engine-byte staging or CUDA Graph capture during timing.
2. CPU preparation: `experiments/prepare_compact_b1.py` (after predecessor exit)
   fits all4families x3seeds on original3000fit positions, preserving2000policy
   calibration positions. No test files opened. No historical artifacts changed.
3. GPU preflight profiles actual memory and cold/warm switch components; define
   safe resident capacities from measurements with reserve, not total6GB alone.
4. `runtime/compact_trt.py` now supplies isolated host-byte/shared-buffer and
   optional CUDA Graph adapters (compiled only, not yet GPU validated). Measured
   integration/launchers, CUDA Graph real-input parity and sequential cache-aware
   calibration still require implementation/verification before launch.
   Source preparation and synthetic tests are NOT completed real measurements.
5. Freeze validation screening and configuration manifest BEFORE held-out tests.
   Preliminary one full10000-image pass per router/ablation; confirm a fixed
   validation-selected shortlist with more repeats. Do not form a huge exhaustive
   Cartesian product by default. All negatives/accuracyfailures are retained.

## Four compact families

- Logistic independent correctness heads, C=.25, standardized12CPUfeatures.
- One shared depth3 regression tree, minleaf20, candidate-correctness leaf vectors.
- ThumbnailMLP: fresh8x8RGB means (192inputs),16hiddenReLU,Kindependent sigmoid
  correctness outputs, Adam .001,20fixedepochs,batch64. Not a CIFARclass predictor.
- Shallow boosted trees:16depth2regressiontrees per nonconstant correctness head,
  fixed regularization; may not be low-overhead whenK=213. Measure rather than
  assuming. Constant all-correct/wrong heads remain constants.

CPU descriptors derive from fresh8x8area means of rawRGB128: RGBmean/std(6),
graymean/std(2),gradientmean/std(2),edgedensity(threshold.1),saturationproxy.
No meaningless between-image brightness variation atbatch1. Runtime must time
fresh thumbnail/features, CPU prediction and host selection. Offline features
are fit/calibration-only. GPU-only NVML energy excludes the CPU cost: do not
claim full-system savings or compare CPUtrainingenergy with GPUtrainingenergy.

## Switching ablations (separate axes, not silently combined)

1. Larger GPU residency: measured safe capacities, report peakVRAM and actual
   candidatesloaded. Keep213logicalchoices vs residentcapacity explicit.
2. Resident subsets: choose complementary pools using fit/validation only, profile
   memory then retain every pool member. CurrentCNN restricted to samepool is a
   required control; otherwise router-size and pool-size effects are confounded.
3. Cache-aware selection: predictedcorrectness minus calibrated penalty for
   measured execution+misscost. Simulate cache sequentially during calibration,
   same initial state and batching as deployment. Never tune against testlabels.
4. Immutable verified enginebytes in hostRAM: verify once at setup, measure setup
   energy/time/RAM; every GPUmiss still includes deserialization/allocation.
5. Allocationcleanup: profile reuse of input/output/activation buffers and less
   forcedGC/emptycache. Enforce measured memorybudget and preserve safe eviction.
6. CUDA Graphs for resident engines only: capture after fixedshape warmup, stable
   buffers/context; actual parity check. Report unsupported configurations, no
   fabricated graph speedup. This does not remove engine-loading misses.

`runtime/compact_switching.py` contains CPU-testable cache-aware selection,
immutable hostbytes and budgetedLRU building blocks; these are NOT yet a complete
GPU adapter. `routers/correctness/compact_cpu.py` contains the4fit/predict families.
All candidates/policies/capacities require a frozen new manifest; no edits to old
timing sources or original weights/reports. No E:,Jetson,smartplug,powermodes,
HPCsubmission orGitHubpush. Exactworker/guardidentity,60sGPUchecks,two>=85Clatch
nextsavedblock/gracefulshutdown,neverrestartlatched;CPUtemperatureunavailable.

Required outputs: perrouter accuracy,CPUfeature/predict latency,total latency,
GPUenergy,loads/evictions,setupcold/warm cost,peakGPU/hostRAM,actualcandidatepool,
accuracyallowance pass,staticcomparators andplots. Trainingenergy/totalsetup
payback must retain scope distinctions; no GPU-only payback for unmeasuredCPU
training. Old baseline monitor remains active at30minutes until both studies
and final audits/plots are genuinely complete.
