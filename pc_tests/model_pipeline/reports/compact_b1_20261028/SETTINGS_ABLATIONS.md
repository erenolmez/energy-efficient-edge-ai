# Additional settings: authorized controlled investigations

These are separate ablations, not retroactive corrections of historical results.
The multi-day batch1 baseline stays cancelled; its8savedfullsweeps remain partial.
Do not construct a full Cartesian product of every setting. Screening uses fit/
validation only; freeze heldoutconfigs first, onefull10000-image preliminarypass
perablation, then a validation-selected confirmation shortlist. Retain failures.

| Question | Fixed comparison | Current status |
|---|---|---|
| Residency restriction | Original engines,three loadingorders,1GiBreserve | Complete/audited:53smallest,12largest,18roundrobin;capacity only |
| Batch1 specialization | Same6fit-selected models,graphs/precision;profile1/32/128 vs1/1/1 | GPU preflight worker launched; not yet an evaluated routing gain |
| Static-cost policy mismatch | Samepool/router/initialcache;staticcost vs measuredexecution+misscost sequential calibration | Selector helpers tested; full calibration/runtime integration pending |
| Expensive eviction | Samechoices/cachecapacity;baselinecleanup vs controlled reuse/no forcedGC | Adapters prepared; memory-growth and runtime tests pending |
| Repeated disk reads/hash | Sameengines/choices;file-load vs immutableverifiedhostbytes | CPUstore tests passed; setupRAM/time/energy andGPUmiss tests pending |
| Host synchronization | CurrentCPUprobabilitycopy vs GPUselection/reducedcopy;CPUcompactrouters separately | Must isolate selection/sync from classifierpool changes; pending |
| GPUlaunch overhead | Same residentengine,fixedshape/buffers;eager vsCUDAgraph | Adapter compiled; realinputcapture/parity/support test pending |
| Cold-start bias | Report setup+coldfirstpass AND warmsteadystate separately | Newruntime must measure both, not hide startup |
| Router data quantity | Fixed seed42 internal stratified fit subsets1000/2000/3000,original2000policycal positions unchanged | Plannedlearningcurve; no freshdata invented |
| Calibration uncertainty | Fixedrouter;predefined500/1000/2000class-balanced calibration subsets and heldoutaccuracybudget outcome | Pending; no test-based penalty/seed selection |
| 20epochs/unweightedBCE | Originalfit only;internaltrain/tune within3000 to compare20/60epoch checkpoints and explicitbalancedloss alternative | Pending; not imply20epochMLPs converged (warnings retained) |
| Batchaggregation | Sameweights/pool;onechoiceperbatch vs independentperimagechoices forsmallpredefinedbatch(8) | Pending; must include partition/scatter/reassembly andswitchcosts; not reusecachedchoices |
| Energy scope | GPUcounter is available; CPU/systemenergy telemetry must be checked, not assumed | GPU-only claims remain; no smartplug/E:/power changes; no full-PC result without valid meter |
| Session/thermal variability | Paired alternating A/B order,sameGPU/software/inputorder;logtemperature,clocks,throttling andpower without changingthem | Paired original/newenginepreflight started; inferenceconfirmation pending |

## Dispatch additions explicitly requested before shutdown

All are QUEUED until human says resume. No GPU work is launched during pause.

1. Current control: GPUrouter ->213probabilities toCPU ->CPUcostselection ->host
   launches independentresidentTRTengine ->GPUclassifier ->hostclassID.
2. GPUcostselection ->onlyint64modelID toCPU ->samehostlaunch/sameclassifier/
   samefinalhostclassID. NativeTorchGPU selector helper prepared withCPU-only
   arithmeticparity tests; realGPUparity/timing/energy remain unmeasured. Device
   constants must be preloaded; no hiddenCPUreadback inside GPUscore calculation.
   HistoricalNumPycostarithmetic isfloat64;FP32fusedselection is separately tested
   for near-tiechoicechanges,not silently substituted.
3. TrueGPUconditional prototype on a smallresidentpool (start with2classifiers),
   conditioncomputedondevice, onlyselectedbranch executes. Compile/import original
   branches,validate actualconditionalboundary/lazyexecution,originalclassifier
   logits/classchoices andresourceusage. `where/select` after executingbothmodels
   is NOT accepted as conditionalrouting. Engine support/build failure is retained
   asunsupported; no speculative performance claim.
4. Finaloutput boundary: hostclassID for allmaincomparisons. A GPUresidentfinal
   output can be a separatelylabelled pipelineexperiment,not compared unfairly
   with anotherpath that still payshostreadback. Keep normalisedimage onGPU in
   all variants;do not insert a GPU->CPUimagecopy.
5. Report routercomputation,selection/readback wait,hostdispatch,classifierexecution
   andtotalwalltime separately;totalGPUenergy pluscoldsetup. Scalarcopy still
   synchronizes,so 852bytes->8bytes reduction is not assumed proportional speedup.

## Gates before held-out evaluation

- Verify12preparedCPUfits and hashes/provenance; serializedbytes measure storage,
  not in-memoryusage, and trainingtime is notCPUenergy.
- Dedicatedengine targets can differ numerically. Report logits/argmaxchanges,
  refresh selectedpoolvalidationpredictions/correctness BEFORE calibrating new
  policies. Never reuse oldcachedtargets as if newengines were identical.
- Load/cost/memoryprofile on validation orsyntheticinputs only. Freeze pool,
  residentcapacity,routermethod,penalty,loss/epochs beforetest access.
- CPUfeatures/descriptors/MLPthumbnail andselection must be computed fresh inside
  runtime timer. Report totalwalllatency, GPUenergy,loads/evictions,CPUdecisiontime,
  coldsetup,warmthroughput,andactualmemory/candidatecounts.
- Accuracyallowance is an aggregate empirical requirement,not per-image/per-class
  assurance. No positivegain claim if actualheldoutaccuracyfails.
- CPU-only router training requires CPUenergy measurement for trainingpayback;
  the earlierGPUfit numerator is valid only for its correspondingTinyCNN.
- Data beyond the original5000routerimages requires a separate clean-data design
  (newreserve/OOF/retraining). It is not authorized implicitly by this ablation.
- No automatic restart of cancelled/failed/latched workers; exactscript/executable/
  creationidentities and freshguard for everyGPUstage. Existing30minmonitor covers
  the authorizedsequence,not a resurrection of the cancelled360-sweepmatrix.
