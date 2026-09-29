# Latency-only router results: 10 seeds

This is a separate **latency-only** experiment. It uses learner seeds
42, 43, 44, 45, 46, 47, 48, 49, 50, 51, the same fixed 2,000 evaluation images, and
the same unpruned `fp32_p0` baseline with 72.75% accuracy. No combined
energy-latency score is used.

| Router | Accuracy | Difference vs p0 | Candidate saving | PC router overhead | Passes |
|---|---:|---:|---:|---:|---:|
| tinycnn8_scalar | 72.39% ± 0.39 | -0.36 pp | 7.29% ± 1.83 | 1.95 ms | 10/10 |
| compact_xgb_shallow | 72.19% ± 0.14 | -0.56 pp | 7.20% ± 1.53 | 2.20 ms | 10/10 |
| tinycnn32_correctness | 72.28% ± 0.38 | -0.47 pp | 7.16% ± 4.20 | 2.10 ms | 10/10 |
| tinycnn32_scalar | 72.34% ± 0.20 | -0.42 pp | 6.05% ± 0.90 | 2.14 ms | 10/10 |
| frozen_mobilenet_xgb_correctness | 72.38% ± 0.27 | -0.37 pp | 5.58% ± 1.43 | 13.33 ms | 10/10 |
| tinycnn16_correctness | 72.67% ± 0.25 | -0.08 pp | 0.78% ± 2.47 | 2.37 ms | 10/10 |
| texture_ridge_advantage | 72.75% ± 0.00 | +0.00 pp | -0.00% ± 0.00 | 1.48 ms | 10/10 |
| compact_knn_correctness | 72.75% ± 0.00 | +0.00 pp | -0.00% ± 0.00 | 2.53 ms | 10/10 |
| tinycnn8_correctness | 72.50% ± 0.45 | -0.25 pp | 3.70% ± 5.97 | 2.27 ms | 9/10 |
| thumbnail_mlp_correctness | 72.58% ± 0.54 | -0.17 pp | 1.52% ± 4.82 | 0.86 ms | 9/10 |
| frozen_mobilenet_xgb_scalar | 72.14% ± 0.34 | -0.61 pp | 12.42% ± 1.59 | 16.85 ms | 8/10 |
| texture_xgb_balanced | 72.08% ± 0.30 | -0.68 pp | 8.03% ± 4.00 | 2.66 ms | 8/10 |
| compact_xgb_correctness | 71.87% ± 0.36 | -0.88 pp | 7.59% ± 1.15 | 1.21 ms | 8/10 |
| enhanced_random_forest_serial | 71.87% ± 0.38 | -0.88 pp | 10.85% ± 5.67 | 15.97 ms | 7/10 |
| texture_top32_xgb | 71.95% ± 0.47 | -0.81 pp | 10.76% ± 4.96 | 2.54 ms | 7/10 |
| tinycnn16_scalar | 72.12% ± 0.56 | -0.62 pp | 8.63% ± 5.37 | 2.05 ms | 7/10 |
| texture_xgb_correctness | 71.88% ± 0.37 | -0.88 pp | 7.66% ± 2.16 | 2.55 ms | 6/10 |
| basic_xgb_scalar | 71.81% ± 0.29 | -0.94 pp | 6.99% ± 1.52 | 1.17 ms | 6/10 |
| enhanced_extra_trees_serial | 71.78% ± 0.28 | -0.97 pp | 14.14% ± 4.43 | 17.93 ms | 5/10 |
| enhanced_xgb_scalar | 71.81% ± 0.19 | -0.94 pp | 9.81% ± 2.79 | 5.41 ms | 5/10 |
| texture_xgb_regularized | 71.70% ± 0.31 | -1.05 pp | 11.94% ± 5.32 | 2.78 ms | 4/10 |
| texture_xgb_advantage | 71.75% ± 0.27 | -1.01 pp | 7.16% ± 1.05 | 1.61 ms | 4/10 |

## Main finding

The largest candidate-latency saving among routers that respected the one-point accuracy budget in all 10 seeds is `tinycnn8_scalar`: 7.29% saving, 72.39% accuracy, and 1.95 ms PC router overhead.

## Average model selections per 2,000 images

| Router | fp32_p0 | fp32_p10 | fp32_p20 | fp32_p30 | fp32_p40 | fp32_p50 | fp32_p60 | fp32_p70 | fp32_p80 | fp32_p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tinycnn8_scalar | 578.7 | 1097.4 | 297.3 | 20.9 | 4.6 | 0.1 | 0.5 | 0.0 | 0.0 | 0.5 |

Counts are averaged across seeds. Each row sums to 2,000 apart from rounding.

The “Passes” column counts seeds whose evaluation accuracy remained within one
percentage point of p0. Candidate latency excludes router latency and is not total system latency.
