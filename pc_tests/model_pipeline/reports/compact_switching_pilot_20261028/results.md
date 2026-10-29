# Validation-only switching pilot

One256image calibration pilot pervariant, fixedorder; timing variation/warmup effects not isolated. Cache-aware policy changes choices, accuracy here is calibration-only. No heldout/fulltest/accuracy-preserving savings claim. Final memory reserve checked every64images; perstep peak not captured.

| Variant | ms/image | GPU J/image | Loads | Evictions |
|---|---:|---:|---:|---:|
| file_static_c2 | 12.406 | 0.3768 | 7 | 5 |
| file_aware_c2 | 5.251 | 0.2702 | 2 | 0 |
| host_static_c2 | 7.471 | 0.3674 | 7 | 5 |
| host_aware_c2 | 4.584 | 0.2590 | 2 | 0 |
| host_static_reuse_c2 | 5.693 | 0.2888 | 7 | 5 |
| resident_static_c6 | 4.907 | 0.2554 | 0 | 0 |
| resident_graph_c6 | 3.562 | 0.2039 | 0 | 0 |
