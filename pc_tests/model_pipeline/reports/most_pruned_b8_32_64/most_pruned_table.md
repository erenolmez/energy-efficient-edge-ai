# Most-pruned static controls: batches8,32,64

Status: COMPLETE; 330/330 sweeps.

21 p90 configurations (7architectures x FP32/FP16/INT8), five10000-image repeats per batch size plus five R101 p0 FP32 references at each size. No router. Seed42+batch shuffle; real tail16 at32/64, no tail at8. Accuracy=repeat mean; latency and GPU-only NVML energy=repeat medians. Savings use the fresh R101 p0 FP32 reference at the SAME batch size, not eligible-static routing comparators. Cold capacity2 engine loading/hash/inference, RGB128 transfer/normalization and sync included; decode/resize/pinning excluded. All failures/negative savings retained. Prior GPU throttling and temporal drift limit strong cross-study latency/energy claims. Batch128 outputs are unchanged.

| Batch | Architecture | Pruning | Precision | Accuracy | Loss pp | Passing allowances | ms/image | mJ/image | L saving % | E saving % |
|---:|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| 8 | efficientnet_b0 | p90 | FP32 | 14.17% | 71.63 | none | 0.407 | 17.812 | +82.58 | +84.30 |
| 8 | efficientnet_b0 | p90 | FP16 | 14.25% | 71.55 | none | 0.353 | 16.713 | +84.88 | +85.27 |
| 8 | efficientnet_b0 | p90 | INT8 | 2.33% | 83.47 | none | 0.551 | 20.749 | +76.39 | +81.71 |
| 8 | mobilenetv3_large | p90 | FP32 | 9.37% | 76.43 | none | 0.278 | 13.841 | +88.10 | +87.80 |
| 8 | mobilenetv3_large | p90 | FP16 | 9.35% | 76.45 | none | 0.249 | 10.923 | +89.34 | +90.37 |
| 8 | mobilenetv3_large | p90 | INT8 | 4.48% | 81.32 | none | 0.356 | 12.233 | +84.74 | +89.22 |
| 8 | mobilenetv3_small | p90 | FP32 | 15.49% | 70.31 | none | 0.249 | 10.217 | +89.32 | +90.99 |
| 8 | mobilenetv3_small | p90 | FP16 | 15.45% | 70.35 | none | 0.236 | 8.636 | +89.88 | +92.39 |
| 8 | mobilenetv3_small | p90 | INT8 | 4.09% | 81.71 | none | 0.330 | 10.975 | +85.87 | +90.33 |
| 8 | resnet101 | p90 | FP32 | 2.21% | 83.59 | none | 0.337 | 16.384 | +85.58 | +85.56 |
| 8 | resnet101 | p90 | FP16 | 2.21% | 83.59 | none | 0.285 | 13.942 | +87.78 | +87.71 |
| 8 | resnet101 | p90 | INT8 | 2.17% | 83.63 | none | 0.369 | 15.101 | +84.18 | +86.69 |
| 8 | resnet18 | p90 | FP32 | 3.45% | 82.35 | none | 0.181 | 8.523 | +92.27 | +92.49 |
| 8 | resnet18 | p90 | FP16 | 3.46% | 82.34 | none | 0.120 | 5.346 | +94.87 | +95.29 |
| 8 | resnet18 | p90 | INT8 | 3.42% | 82.38 | none | 0.117 | 5.576 | +95.00 | +95.08 |
| 8 | resnet34 | p90 | FP32 | 3.19% | 82.61 | none | 0.218 | 10.055 | +90.67 | +91.14 |
| 8 | resnet34 | p90 | FP16 | 3.18% | 82.62 | none | 0.146 | 6.973 | +93.74 | +93.85 |
| 8 | resnet34 | p90 | INT8 | 3.31% | 82.49 | none | 0.187 | 8.076 | +92.01 | +92.88 |
| 8 | resnet50 | p90 | FP32 | 3.82% | 81.98 | none | 0.250 | 10.795 | +89.31 | +90.48 |
| 8 | resnet50 | p90 | FP16 | 3.86% | 81.94 | none | 0.185 | 8.203 | +92.07 | +92.77 |
| 8 | resnet50 | p90 | INT8 | 2.77% | 83.03 | none | 0.223 | 9.715 | +90.45 | +91.44 |
| 32 | efficientnet_b0 | p90 | FP32 | 14.17% | 71.63 | none | 0.135 | 6.530 | +92.81 | +93.03 |
| 32 | efficientnet_b0 | p90 | FP16 | 14.25% | 71.55 | none | 0.105 | 4.613 | +94.43 | +95.08 |
| 32 | efficientnet_b0 | p90 | INT8 | 2.33% | 83.47 | none | 0.149 | 6.442 | +92.05 | +93.12 |
| 32 | mobilenetv3_large | p90 | FP32 | 9.37% | 76.43 | none | 0.102 | 4.655 | +94.59 | +95.03 |
| 32 | mobilenetv3_large | p90 | FP16 | 9.35% | 76.45 | none | 0.085 | 3.687 | +95.47 | +96.06 |
| 32 | mobilenetv3_large | p90 | INT8 | 4.48% | 81.32 | none | 0.093 | 3.921 | +95.03 | +95.81 |
| 32 | mobilenetv3_small | p90 | FP32 | 15.49% | 70.31 | none | 0.076 | 3.120 | +95.98 | +96.67 |
| 32 | mobilenetv3_small | p90 | FP16 | 15.45% | 70.35 | none | 0.072 | 3.327 | +96.16 | +96.45 |
| 32 | mobilenetv3_small | p90 | INT8 | 4.09% | 81.71 | none | 0.091 | 3.856 | +95.14 | +95.88 |
| 32 | resnet101 | p90 | FP32 | 2.21% | 83.59 | none | 0.194 | 8.741 | +89.66 | +90.67 |
| 32 | resnet101 | p90 | FP16 | 2.21% | 83.59 | none | 0.102 | 5.130 | +94.60 | +94.52 |
| 32 | resnet101 | p90 | INT8 | 2.17% | 83.63 | none | 0.112 | 4.953 | +94.05 | +94.71 |
| 32 | resnet18 | p90 | FP32 | 3.45% | 82.35 | none | 0.124 | 5.963 | +93.40 | +93.63 |
| 32 | resnet18 | p90 | FP16 | 3.46% | 82.34 | none | 0.069 | 3.738 | +96.33 | +96.01 |
| 32 | resnet18 | p90 | INT8 | 3.42% | 82.38 | none | 0.056 | 2.646 | +97.02 | +97.17 |
| 32 | resnet34 | p90 | FP32 | 3.19% | 82.61 | none | 0.139 | 6.439 | +92.62 | +93.13 |
| 32 | resnet34 | p90 | FP16 | 3.18% | 82.62 | none | 0.076 | 3.583 | +95.97 | +96.17 |
| 32 | resnet34 | p90 | INT8 | 3.31% | 82.49 | none | 0.072 | 3.409 | +96.18 | +96.36 |
| 32 | resnet50 | p90 | FP32 | 3.82% | 81.98 | none | 0.146 | 7.451 | +92.26 | +92.04 |
| 32 | resnet50 | p90 | FP16 | 3.86% | 81.94 | none | 0.087 | 4.345 | +95.39 | +95.36 |
| 32 | resnet50 | p90 | INT8 | 2.77% | 83.03 | none | 0.082 | 3.462 | +95.65 | +96.30 |
| 64 | efficientnet_b0 | p90 | FP32 | 14.17% | 71.63 | none | 0.124 | 5.032 | +93.41 | +94.58 |
| 64 | efficientnet_b0 | p90 | FP16 | 14.25% | 71.55 | none | 0.091 | 3.559 | +95.15 | +96.16 |
| 64 | efficientnet_b0 | p90 | INT8 | 2.33% | 83.47 | none | 0.109 | 4.127 | +94.21 | +95.55 |
| 64 | mobilenetv3_large | p90 | FP32 | 9.37% | 76.43 | none | 0.096 | 3.862 | +94.92 | +95.84 |
| 64 | mobilenetv3_large | p90 | FP16 | 9.35% | 76.45 | none | 0.069 | 2.728 | +96.31 | +97.06 |
| 64 | mobilenetv3_large | p90 | INT8 | 4.48% | 81.32 | none | 0.075 | 2.592 | +96.02 | +97.21 |
| 64 | mobilenetv3_small | p90 | FP32 | 15.49% | 70.31 | none | 0.061 | 2.205 | +96.77 | +97.62 |
| 64 | mobilenetv3_small | p90 | FP16 | 15.45% | 70.35 | none | 0.054 | 1.779 | +97.12 | +98.08 |
| 64 | mobilenetv3_small | p90 | INT8 | 4.09% | 81.71 | none | 0.061 | 2.001 | +96.77 | +97.84 |
| 64 | resnet101 | p90 | FP32 | 2.21% | 83.59 | none | 0.183 | 8.030 | +90.26 | +91.34 |
| 64 | resnet101 | p90 | FP16 | 2.21% | 83.59 | none | 0.103 | 4.001 | +94.51 | +95.69 |
| 64 | resnet101 | p90 | INT8 | 2.17% | 83.63 | none | 0.094 | 3.390 | +95.01 | +96.35 |
| 64 | resnet18 | p90 | FP32 | 3.45% | 82.35 | none | 0.126 | 5.972 | +93.28 | +93.56 |
| 64 | resnet18 | p90 | FP16 | 3.46% | 82.34 | none | 0.075 | 2.872 | +96.02 | +96.90 |
| 64 | resnet18 | p90 | INT8 | 3.42% | 82.38 | none | 0.057 | 2.300 | +96.97 | +97.52 |
| 64 | resnet34 | p90 | FP32 | 3.19% | 82.61 | none | 0.136 | 6.156 | +92.78 | +93.36 |
| 64 | resnet34 | p90 | FP16 | 3.18% | 82.62 | none | 0.073 | 2.819 | +96.11 | +96.96 |
| 64 | resnet34 | p90 | INT8 | 3.31% | 82.49 | none | 0.065 | 2.289 | +96.53 | +97.53 |
| 64 | resnet50 | p90 | FP32 | 3.82% | 81.98 | none | 0.148 | 6.886 | +92.11 | +92.58 |
| 64 | resnet50 | p90 | FP16 | 3.86% | 81.94 | none | 0.087 | 3.600 | +95.38 | +96.12 |
| 64 | resnet50 | p90 | INT8 | 2.77% | 83.03 | none | 0.069 | 2.700 | +96.31 | +97.09 |
