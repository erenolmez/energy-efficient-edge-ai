# Most-pruned static controls: batch128

Status: COMPLETE; 110/110 sweeps saved.

Each architecture uses its highest stored pruning level (p90), with FP32/FP16/INT8; ShuffleNet has no pruned variants and is excluded. No router. Five repeats of 10000 held-out images; common seed42+128 batch ordering; real tail16; same RGB128 timing boundary and cold cache as the random study. Accuracy is the repeat mean, latency/energy the repeat medians. Savings below are versus freshly measured ResNet101 p0 FP32, NOT the eligible-static comparator in the routing matrix. Negative savings and all accuracy failures are retained. GPU-only NVML energy. These measurements follow the random study in time; cross-study timing comparisons remain subject to previously observed throttling and temporal drift.

| Architecture | Pruning | Precision | Accuracy | Loss pp | Pass allowances (pp) | ms/image | mJ/image | L saving vs R101 FP32 | E saving vs R101 FP32 |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| efficientnet_b0 | p90 | FP32 | 14.17% | 71.63 | none | 0.101 | 4.559 | +94.11% | +94.82% |
| efficientnet_b0 | p90 | FP16 | 14.25% | 71.55 | none | 0.070 | 3.025 | +95.88% | +96.56% |
| efficientnet_b0 | p90 | INT8 | 2.33% | 83.47 | none | 0.078 | 3.323 | +95.42% | +96.23% |
| mobilenetv3_large | p90 | FP32 | 9.37% | 76.43 | none | 0.079 | 3.567 | +95.37% | +95.95% |
| mobilenetv3_large | p90 | FP16 | 9.35% | 76.45 | none | 0.054 | 2.492 | +96.82% | +97.17% |
| mobilenetv3_large | p90 | INT8 | 4.48% | 81.32 | none | 0.054 | 2.334 | +96.82% | +97.35% |
| mobilenetv3_small | p90 | FP32 | 15.49% | 70.31 | none | 0.047 | 1.925 | +97.26% | +97.81% |
| mobilenetv3_small | p90 | FP16 | 15.45% | 70.35 | none | 0.041 | 1.813 | +97.62% | +97.94% |
| mobilenetv3_small | p90 | INT8 | 4.09% | 81.71 | none | 0.041 | 1.476 | +97.58% | +98.32% |
| resnet101 | p90 | FP32 | 2.21% | 83.59 | none | 0.154 | 7.472 | +90.99% | +91.51% |
| resnet101 | p90 | FP16 | 2.21% | 83.59 | none | 0.085 | 3.678 | +95.04% | +95.82% |
| resnet101 | p90 | INT8 | 2.17% | 83.63 | none | 0.071 | 2.994 | +95.86% | +96.60% |
| resnet18 | p90 | FP32 | 3.45% | 82.35 | none | 0.112 | 5.341 | +93.47% | +93.93% |
| resnet18 | p90 | FP16 | 3.46% | 82.34 | none | 0.064 | 2.965 | +96.28% | +96.63% |
| resnet18 | p90 | INT8 | 3.42% | 82.38 | none | 0.047 | 2.022 | +97.23% | +97.70% |
| resnet34 | p90 | FP32 | 3.19% | 82.61 | none | 0.116 | 5.743 | +93.19% | +93.48% |
| resnet34 | p90 | FP16 | 3.18% | 82.62 | none | 0.060 | 2.590 | +96.47% | +97.06% |
| resnet34 | p90 | INT8 | 3.31% | 82.49 | none | 0.054 | 2.050 | +96.85% | +97.67% |
| resnet50 | p90 | FP32 | 3.82% | 81.98 | none | 0.128 | 6.337 | +92.53% | +92.80% |
| resnet50 | p90 | FP16 | 3.86% | 81.94 | none | 0.073 | 3.365 | +95.74% | +96.18% |
| resnet50 | p90 | INT8 | 2.77% | 83.03 | none | 0.057 | 2.481 | +96.70% | +97.18% |

Validation-only eligible choices among these21 (selected before test evaluation):

```json
{
  "42_latency_0.5": null,
  "42_latency_1": null,
  "42_latency_1.5": null,
  "42_latency_2": null,
  "42_energy_0.5": null,
  "42_energy_1": null,
  "42_energy_1.5": null,
  "42_energy_2": null,
  "45_latency_0.5": null,
  "45_latency_1": null,
  "45_latency_1.5": null,
  "45_latency_2": null,
  "45_energy_0.5": null,
  "45_energy_1": null,
  "45_energy_1.5": null,
  "45_energy_2": null,
  "48_latency_0.5": null,
  "48_latency_1": null,
  "48_latency_1.5": null,
  "48_latency_2": null,
  "48_energy_0.5": null,
  "48_energy_1": null,
  "48_energy_1.5": null,
  "48_energy_2": null
}
```
