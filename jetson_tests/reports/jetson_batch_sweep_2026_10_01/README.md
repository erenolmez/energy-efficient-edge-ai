# Jetson batch-size and difficulty-percentile routing sweep

## Question

Can a TinyCNN route a complete CIFAR-100 batch to one of the distilled FP32
ResNet-18 models (`p0`, `p30`, `p50`, or `p70`) while staying within 0.5
percentage points of the unpruned `p0` accuracy and reducing either measured
latency or measured VDD_IN energy?

## Experimental design

- Every system received the same fixed 2,000 CIFAR-100 test images.
- The direct comparison baseline was always the unpruned FP32 `p0` model at
  the same batch size. No oracle result was used as a reported baseline.
- Batch sizes were 8, 16, 32, 64, and 128.
- A single decision was made for each batch from the 50th, 70th, 80th, or 90th
  percentile of its per-image difficulty scores.
- Three independently trained TinyCNN routers (seeds 45, 48, and 51) were
  evaluated. Each Jetson measurement used three complete repeats.
- The energy experiment used the 624.75 MHz GPU profile. The latency experiment
  used 612 MHz. Every TensorRT engine was rebuilt under its measurement profile;
  the first mismatched latency run was discarded.
- Energy means total-module VDD_IN energy per image, not GPU power alone.
- A route is accuracy-valid when its accuracy is no more than 0.5 percentage
  points below the measured `p0` result.

The full 10,000-image PyTorch test-set accuracies reproduce the earlier model
results:

| Model | Full-test accuracy |
| --- | ---: |
| `fp32_p0` | 73.70% |
| `fp32_p30` | 71.51% |
| `fp32_p50` | 69.89% |
| `fp32_p70` | 66.09% |

The routing subset gives `p0` 72.55%. The difference from 73.70% is expected
because routing uses a fixed 2,000-image subset rather than all 10,000 images.

## Static model result

All four candidates are Pareto-efficient: stronger pruning lowers latency and
energy but also lowers accuracy. At batch 128:

| Profile | Model | Accuracy | Latency (ms/image) | VDD_IN (mJ/image) |
| --- | --- | ---: | ---: | ---: |
| 624.75 MHz | `p0` | 72.55% | 1.545 | 15.051 |
| 624.75 MHz | `p30` | 71.00% | 1.080 | 9.948 |
| 624.75 MHz | `p50` | 68.95% | 0.772 | 6.720 |
| 624.75 MHz | `p70` | 66.55% | 0.575 | 4.589 |
| 612 MHz | `p0` | 72.55% | 1.547 | 13.848 |
| 612 MHz | `p30` | 71.00% | 1.098 | 9.278 |
| 612 MHz | `p50` | 68.95% | 0.783 | 6.297 |
| 612 MHz | `p70` | 66.55% | 0.577 | 4.318 |

Static `p30` therefore saves about 29% latency and 33% energy at batch 128,
but loses 1.55 accuracy points on this subset and does not satisfy the strict
0.5-point deployment constraint.

## Dynamic routing results

Best accuracy-valid energy policy at each batch size:

| Batch | Percentile/router | Accuracy delta | Energy saving | Latency saving | Router ms/image |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | P50 / seed 51 | +0.10 pp | -22.01% | -32.62% | 0.797 |
| 16 | P90 / seed 45 | 0.00 pp | -17.01% | -24.67% | 0.451 |
| 32 | P50 / seed 51 | 0.00 pp | -9.51% | -14.04% | 0.249 |
| 64 | P50 / seed 51 | 0.00 pp | -5.95% | -9.09% | 0.150 |
| 128 | P80 / seed 48 | 0.00 pp | -2.91% | -5.06% | 0.090 |

Best accuracy-valid latency policy at each batch size:

| Batch | Percentile/router | Accuracy delta | Latency saving | Energy saving | Router ms/image |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | P70 / seed 51 | -0.10 pp | -30.95% | -22.65% | 0.765 |
| 16 | P50 / seed 51 | 0.00 pp | -24.88% | -16.69% | 0.481 |
| 32 | P90 / seed 48 | 0.00 pp | -14.52% | -9.73% | 0.261 |
| 64 | P90 / seed 51 | 0.00 pp | -8.86% | -5.92% | 0.153 |
| 128 | P80 / seed 48 | 0.00 pp | -5.67% | -3.62% | 0.091 |

Negative saving means the routed system consumed more time or energy than the
direct `p0` baseline. Larger batches amortize TinyCNN overhead, but none of the
accuracy-valid configurations produces a positive measured saving.

## Which models were selected?

The best energy route at batch 8 selected `p50` for 13 of 250 batches (104
images) and `p0` for the remaining 237 batches (1,896 images). The best latency
route at batch 8 selected `p50` once (8 images); its batch-16 route selected
`p50` once (16 images). The best valid routes at larger batches selected only
`p0`. No best valid route selected `p30` or `p70`.

This explains the negative outcome: the strict accuracy limit makes the batch
decision conservative, while the TinyCNN cost is paid for every batch even when
the final choice is `p0`.

## Conclusion and next experiment

The current batch router is **not deployment-beneficial** under a 0.5-point
accuracy budget. This is a useful negative result: it separates the large
potential savings of the pruned models from the overhead and accuracy behavior
of the actual routing system.

The next experiment should train a router directly on batch-level targets,
measure a TensorRT-optimized TinyCNN, and include less aggressively pruned
models such as `p10` and `p20`. Lower difficulty percentiles (for example P10
and P25) should also be tested because P50-P90 strongly biases a batch toward
`p0`. A relaxed accuracy budget can be reported as a separate trade-off curve,
but should not replace the strict result above.

## Files

- `static_energy_summary.csv` and `static_latency_summary.csv`: every static
  model, batch size, repeat summary, and Pareto flag.
- `energy_baseline_comparison.csv` and `latency_baseline_comparison.csv`: all
  measured router/baseline comparisons.
- `energy_best_by_batch.csv` and `latency_best_by_batch.csv`: best valid route
  for each batch size.
- `*_best_by_batch_percentile.csv`: best valid router for every batch and
  percentile.
- `best_router_savings_vs_p0.png`: measured saving relative to direct `p0`.
- `best_router_model_counts.png`: selected model counts for the best routes.
- `*_p*_system_comparison.png` and `*_p*_model_selection_counts.png`: complete
  per-percentile plots.
- `tinycnn_training_*.csv`: PC-side router calibration summaries.
- `full_test_model_accuracy.csv`: full 10,000-image accuracy verification.
