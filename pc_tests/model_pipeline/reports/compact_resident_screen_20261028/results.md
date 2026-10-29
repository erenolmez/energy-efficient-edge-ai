# Preliminary batch-one resident comparison

One-repeat preliminary latency1pp same6originalresidentpool, not213choice/capacity2. No test tuning. No seed selection.

Accuracy pass means loss no more than1percentage point versus the same-seed fresh R101FP32 reference. Savings below use the fresh validation-selected static comparator; positive means lower latency/energy. All21sweeps use10000testimages.

| Seed | Method | Accuracy % | Loss pp | Pass | ms/image | Latency saving % | GPUenergy saving % |
|---:|---|---:|---:|:---:|---:|---:|---:|
| 42 | tree_d3 | 84.87 | 0.93 | yes | 4.123 | +5.59 | -4.41 |
| 42 | logistic | 84.62 | 1.18 | NO | 5.429 | -24.32 | -33.98 |
| 42 | tinycnn | 84.91 | 0.89 | yes | 4.154 | +4.87 | -8.28 |
| 42 | thumbnail_mlp16 | 84.98 | 0.82 | yes | 4.445 | -1.79 | -1.56 |
| 42 | eligible_static | 85.85 | -0.05 | yes | 4.367 | +0.00 | +0.00 |
| 42 | boosted_d2 | 84.43 | 1.37 | NO | 6.593 | -50.97 | -40.49 |
| 42 | reference | 85.80 | 0.00 | yes | 8.169 | -87.06 | -60.15 |
| 45 | logistic | 84.17 | 1.63 | NO | 5.892 | -20.45 | -25.52 |
| 45 | tinycnn | 84.55 | 1.25 | NO | 4.402 | +10.01 | +8.56 |
| 45 | boosted_d2 | 85.01 | 0.79 | yes | 7.644 | -56.27 | -48.56 |
| 45 | reference | 85.80 | 0.00 | yes | 8.967 | -83.32 | -77.56 |
| 45 | eligible_static | 85.85 | -0.05 | yes | 4.891 | +0.00 | +0.00 |
| 45 | tree_d3 | 85.02 | 0.78 | yes | 4.980 | -1.82 | +3.49 |
| 45 | thumbnail_mlp16 | 84.64 | 1.16 | NO | 5.164 | -5.57 | -0.20 |
| 48 | thumbnail_mlp16 | 84.27 | 1.53 | NO | 4.473 | +12.27 | +10.42 |
| 48 | tinycnn | 84.50 | 1.30 | NO | 4.462 | +12.50 | +10.56 |
| 48 | logistic | 84.01 | 1.79 | NO | 5.759 | -12.95 | -15.42 |
| 48 | tree_d3 | 83.89 | 1.91 | NO | 4.546 | +10.85 | +8.92 |
| 48 | boosted_d2 | 84.19 | 1.61 | NO | 6.646 | -30.34 | -28.77 |
| 48 | reference | 85.80 | 0.00 | yes | 9.247 | -81.35 | -82.37 |
| 48 | eligible_static | 85.85 | -0.05 | yes | 5.099 | +0.00 | +0.00 |

No repeated-sweep uncertainty estimate. Static references sampled in same session but not temporally interleaved perimage. Observed performance variation may affect small savings; preserve negatives. No final optimized routing claim.

TinyCNN only, seed-specific measured GPU fit energy /positive matched warm inference saving; accuracy gate. CPU training energy unavailable. Original fit/new inference sessions, no full-PC/completepipeline payback.

The four compact routers are CPU implementations: GPU-only energy does not include their CPU energy. Independent logistic/boosted heads evaluate only6available candidates; sharedMLP/tree/CNN retain full213outputcompute. Startup pool loading is separate from warm inference.
