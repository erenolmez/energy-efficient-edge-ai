# Six-model switch counts and Jetson8GB PC simulation

Completed CPU-only analysis of102 saved10,000-request sweeps. No model inference, images/labels, Jetson/SSH or hardware access.

A switch means a different model on the next request; cold first load is not a switch. A model switch is not necessarily a cache miss.

Choices are reordered using each saved execution-order permutation. All trace/report hashes and selection totals checked; original capacity2 LRU load/eviction counts reproduced exactly.

## Study A: compact routers, all six engines resident

| Router | Seed42 switches | Seed45 switches | Seed48 switches |
|---|---:|---:|---:|
| boosted_d2 | 4979 | 3760 | 5063 |
| eligible_static | 0 | 0 | 0 |
| logistic | 4783 | 5033 | 5337 |
| reference | 0 | 0 | 0 |
| thumbnail_mlp16 | 4136 | 5910 | 5129 |
| tinycnn | 3936 | 5010 | 5207 |
| tree_d3 | 4475 | 4411 | 5853 |

## Study B: TinyCNN switching confirmation

Ranges span three repeats per seed; no pooling as independent observations.

| Variant | Seed42 switch range | Seed45 switch range | Seed48 switch range |
|---|---:|---:|---:|
| eligible_eager | 0 | 0 | 0 |
| eligible_graph | 0 | 0 | 0 |
| file_static_c2 | 3936 | 5010 | 5207 |
| host_aware_c2 | 0 | 0 | 0 |
| host_static_reuse_c2 | 3936 | 5010 | 5207 |
| reference_eager | 0 | 0 | 0 |
| reference_graph | 0 | 0 | 0 |
| resident_eager_c6 | 3936 | 5010 | 5207 |
| resident_graph_c6 | 3936 | 5010 | 5207 |

## Directional transitions: Study A TinyCNN, seed42

| From | To | Switches |
|---|---|---:|
| resnet101__p0__fp16 | efficientnet_b0__p0__fp16 | 1782 |
| resnet101__p0__fp16 | resnet34__p0__int8 | 103 |
| resnet101__p0__fp16 | resnet18__p10__fp16 | 27 |
| efficientnet_b0__p0__fp16 | resnet101__p0__fp16 | 1770 |
| efficientnet_b0__p0__fp16 | resnet34__p0__int8 | 47 |
| efficientnet_b0__p0__fp16 | resnet18__p10__fp16 | 15 |
| resnet34__p0__int8 | resnet101__p0__fp16 | 110 |
| resnet34__p0__int8 | efficientnet_b0__p0__fp16 | 40 |
| resnet18__p10__fp16 | resnet101__p0__fp16 | 33 |
| resnet18__p10__fp16 | efficientnet_b0__p0__fp16 | 9 |

## TinyCNN selected destinations (images, not switches)

| Model | Seed42 | Seed45 | Seed48 |
|---|---:|---:|---:|
| resnet101__p0__fp32 | 0 | 0 | 0 |
| resnet101__p0__fp16 | 7363 | 5685 | 5903 |
| efficientnet_b0__p0__fp16 | 2442 | 4130 | 3713 |
| resnet34__p0__int8 | 153 | 98 | 0 |
| mobilenetv3_large__p10__fp16 | 0 | 0 | 352 |
| resnet18__p10__fp16 | 42 | 87 | 32 |

Full exact directional counts (including zeros) for every router/seed/repeat are in from_to_transitions.csv.

## PC-based Jetson8GB simulation

Fixed saved policy traces; byte-LRU scenarios, NOT Jetson hardware/time/energy emulation. Cache-aware policies are NOT recalibrated for new caches.

8e9-byte budget;2/3/4GiB OS+applications,1GiB safety,0.5GiB runtime/router; original host-store bytes and128-image raw+twoFP32 buffers subtracted. Allocation deltas multiplied by1/1.5/2; capacity2 versus up-to6 LRU. Cold-start loads included. No timing/energy predictions.

Central scenario across102 traces: loads range 1-5, evictions range 0-0, infeasible requests 0. Demand-loaded cache, NOT preloading all six. Not a measured Jetson result.

## Qualifications

- Repeated sweeps are not independent input samples; retain per-seed/per-repeat results.
- Historical held-out decision traces are descriptive only; no test-based policy selection or tuning.
- Memory multipliers/reserves are assumptions, not confidence bounds or verified Jetson capacities.
- Serialized host store, CPU+GPU allocations share the same assumed budget; build/load transient peaks unmeasured.
