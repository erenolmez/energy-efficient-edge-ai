# Energy-only router results: 10 seeds

This is a separate **energy-only** experiment. It uses learner seeds
42, 43, 44, 45, 46, 47, 48, 49, 50, 51, the same fixed 2,000 evaluation images, and
the same unpruned `fp32_p0` baseline with 72.75% accuracy. No combined
energy-latency score is used.

| Router | Accuracy | Difference vs p0 | Candidate saving | PC router overhead | Passes |
|---|---:|---:|---:|---:|---:|
| frozen_mobilenet_xgb_scalar | 72.17% ± 0.17 | -0.57 pp | 14.24% ± 2.61 | 17.28 ms | 10/10 |
| frozen_mobilenet_xgb_correctness | 72.27% ± 0.27 | -0.48 pp | 8.05% ± 2.85 | 13.13 ms | 10/10 |
| texture_xgb_balanced | 72.06% ± 0.21 | -0.69 pp | 7.76% ± 1.43 | 2.92 ms | 10/10 |
| compact_xgb_shallow | 72.41% ± 0.15 | -0.35 pp | 7.02% ± 0.38 | 2.30 ms | 10/10 |
| tinycnn32_scalar | 72.32% ± 0.32 | -0.43 pp | 6.98% ± 1.66 | 2.11 ms | 10/10 |
| tinycnn16_correctness | 72.67% ± 0.25 | -0.08 pp | 0.88% ± 2.80 | 2.15 ms | 10/10 |
| texture_ridge_advantage | 72.75% ± 0.00 | +0.00 pp | 0.00% ± 0.00 | 1.47 ms | 10/10 |
| compact_knn_correctness | 72.75% ± 0.00 | +0.00 pp | 0.00% ± 0.00 | 2.54 ms | 10/10 |
| tinycnn32_correctness | 72.27% ± 0.37 | -0.48 pp | 8.37% ± 5.07 | 2.31 ms | 9/10 |
| basic_xgb_scalar | 72.08% ± 0.31 | -0.68 pp | 8.22% ± 2.76 | 1.38 ms | 9/10 |
| tinycnn8_correctness | 72.51% ± 0.44 | -0.24 pp | 4.24% ± 6.83 | 2.05 ms | 9/10 |
| thumbnail_mlp_correctness | 72.57% ± 0.57 | -0.18 pp | 1.76% ± 5.57 | 0.89 ms | 9/10 |
| tinycnn8_scalar | 72.05% ± 0.37 | -0.70 pp | 10.23% ± 3.44 | 1.97 ms | 8/10 |
| tinycnn16_scalar | 71.98% ± 0.56 | -0.77 pp | 12.56% ± 5.42 | 2.05 ms | 7/10 |
| texture_xgb_correctness | 71.87% ± 0.35 | -0.88 pp | 9.70% ± 3.18 | 2.51 ms | 6/10 |
| texture_xgb_advantage | 71.72% ± 0.23 | -1.03 pp | 9.55% ± 1.57 | 1.60 ms | 6/10 |
| compact_xgb_correctness | 71.91% ± 0.39 | -0.84 pp | 9.43% ± 1.85 | 1.20 ms | 6/10 |
| texture_top32_xgb | 71.94% ± 0.61 | -0.82 pp | 12.49% ± 5.90 | 2.71 ms | 5/10 |
| enhanced_xgb_scalar | 71.81% ± 0.34 | -0.94 pp | 10.98% ± 2.42 | 5.64 ms | 5/10 |
| enhanced_random_forest_serial | 71.62% ± 0.36 | -1.13 pp | 17.37% ± 6.02 | 16.40 ms | 3/10 |
| texture_xgb_regularized | 71.50% ± 0.35 | -1.24 pp | 16.97% ± 4.57 | 2.93 ms | 3/10 |
| enhanced_extra_trees_serial | 71.58% ± 0.17 | -1.18 pp | 20.49% ± 0.91 | 18.87 ms | 2/10 |

## Main finding

The largest candidate-energy saving among routers that respected the one-point accuracy budget in all 10 seeds is `frozen_mobilenet_xgb_scalar`: 14.24% saving, 72.17% accuracy, and 17.28 ms PC router overhead. Restricting attention to routers below 5 ms PC overhead, `texture_xgb_balanced` has the largest stable saving: 7.76% at 72.06% accuracy.

## Average model selections per 2,000 images

| Router | fp32_p0 | fp32_p10 | fp32_p20 | fp32_p30 | fp32_p40 | fp32_p50 | fp32_p60 | fp32_p70 | fp32_p80 | fp32_p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen_mobilenet_xgb_scalar | 302.6 | 723.2 | 658.5 | 265.2 | 47.0 | 0.1 | 0.0 | 0.0 | 0.0 | 3.4 |
| texture_xgb_balanced | 646.0 | 1066.9 | 259.6 | 27.1 | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Counts are averaged across seeds. Each row sums to 2,000 apart from rounding.

The “Passes” column counts seeds whose evaluation accuracy remained within one
percentage point of p0. Candidate energy excludes router energy and is not total system energy.
