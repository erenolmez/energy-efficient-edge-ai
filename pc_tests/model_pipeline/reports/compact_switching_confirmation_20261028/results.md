# Batch-one switching confirmation

Three repeats, all3seeds, six original candidates, latency1pp. Mode-matched eager/graph controls; no test-based tuning. Compare only matchingpool/configuration.

| Seed | Variant | Accuracy % | Loss pp | Pass | ms/image | Latency saving % | GPU-energy saving % |
|---:|---|---:|---:|:---:|---:|---:|---:|
| 42 | file_static_c2 | 84.91 | 0.89 | yes | 15.914 | -192.38 | -181.39 |
| 42 | host_static_reuse_c2 | 84.91 | 0.89 | yes | 7.901 | -45.16 | -37.53 |
| 42 | host_aware_c2 | 85.85 | -0.05 | yes | 6.022 | -10.64 | -6.45 |
| 42 | resident_eager_c6 | 84.91 | 0.89 | yes | 6.791 | -24.78 | -80.44 |
| 42 | resident_graph_c6 | 84.91 | 0.89 | yes | 6.351 | +9.07 | +4.20 |
| 45 | file_static_c2 | 84.55 | 1.25 | NO | 16.587 | -77.96 | -94.88 |
| 45 | host_static_reuse_c2 | 84.55 | 1.25 | NO | 7.611 | +18.35 | +10.16 |
| 45 | host_aware_c2 | 85.85 | -0.05 | yes | 6.689 | +28.23 | +22.12 |
| 45 | resident_eager_c6 | 84.55 | 1.25 | NO | 6.657 | +28.58 | +27.42 |
| 45 | resident_graph_c6 | 84.55 | 1.25 | NO | 6.424 | +11.68 | +8.11 |
| 48 | file_static_c2 | 84.50 | 1.30 | NO | 23.149 | -299.88 | -315.81 |
| 48 | host_static_reuse_c2 | 84.50 | 1.30 | NO | 8.203 | -41.69 | -46.99 |
| 48 | host_aware_c2 | 85.85 | -0.05 | yes | 6.938 | -19.85 | -14.54 |
| 48 | resident_eager_c6 | 84.50 | 1.30 | NO | 7.725 | -33.45 | -21.18 |
| 48 | resident_graph_c6 | 84.50 | 1.30 | NO | 6.070 | +18.23 | +15.53 |

Savings use fresh same-mode eligible-static comparators; positive means lower cost. Reference-FP32 comparisons andseed-specific fit payback are inJSON.

Seed-specific TinyCNN GPUfit-only numerator, positive matchedGPUinferencesaving denominator, allrepeataccuracygate. Excludes CPU,supervision,calibration,export,startup. Training/inference separate sessions; measurements straddle saved recovery, notwholePC energy.

Three repeats are descriptive ranges, not confidence intervals. Sequential caseorders/sessionthermals may affect differences: eligible-eager timings varied substantially across repeats/sessions. All host-aware cases collapsed to the same staticR101FP16; faster same-classifier measurements are not proof of dynamic-selection benefit. Keep rawcoldloads/setup separate; resident warm inference excludes pool startup.

All41pre-repair fullrows and5archivedpartialchunks preserved. Sourcehashlineage/frozenplan andactualpredictions checked.
