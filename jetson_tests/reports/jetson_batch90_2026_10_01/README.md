# 90th-percentile batch-routing results

This follow-up evaluates one routing decision per batch on the same fixed 2,000
CIFAR-100 test images. Every image receives a difficulty score, the 90th
percentile becomes the batch score, and every image in that batch is sent to
the same TensorRT FP32 ResNet-18 variant. Batch sizes 8, 16, and 32 were tested
with three complete repeats. Each batch size has its own unpruned `fp32_p0`
baseline, so all percentages below are like-for-like comparisons.

Two experiments remain separate:

- energy objective: 15 W nvpmodel mode with the GPU fixed at 624.75 MHz;
- latency objective: 15 W nvpmodel mode with the GPU fixed at 612 MHz.

VDD_IN energy includes the router, routing decision, preprocessing, selected
TensorRT model, synchronization, and the complete measured interval.

## Main result

The 90th-percentile rule preserves accuracy, but it does **not** reduce total
latency or energy with the present routers. Larger batches make the rule more
conservative: a batch is increasingly likely to contain at least one difficult
image, so most or all batches are sent to p0. The router then becomes pure
overhead.

### Energy-objective sweep

| Batch | p0 baseline | Best routed system by energy | Accuracy delta | Energy change vs p0 | Latency change vs p0 |
|---:|---:|---|---:|---:|---:|
| 8 | 20.485 mJ/image | compact XGBoost s48: 29.754 mJ/image | -0.15 pp | **+45.25%** | **+72.10%** |
| 16 | 16.285 mJ/image | compact XGBoost s51: 25.434 mJ/image | +0.05 pp | **+56.17%** | **+88.75%** |
| 32 | 14.811 mJ/image | compact XGBoost s48: 23.587 mJ/image | 0.00 pp | **+59.25%** | **+98.04%** |

The handcrafted feature extractors cost roughly 1.45--2.35 ms/image. At batch
32 the compact s48 policy selected p0 for all 63 batches, so its 59.25% energy
penalty came entirely from routing overhead.

### Latency-objective sweep

| Batch | p0 baseline | Best routed system by latency | Accuracy delta | Latency change vs p0 | Energy change vs p0 |
|---:|---:|---|---:|---:|---:|
| 8 | 2.580 ms/image | TinyCNN s51: 3.124 ms/image | +0.35 pp | **+21.11%** | **+13.28%** |
| 16 | 1.954 ms/image | TinyCNN s51: 2.318 ms/image | +0.15 pp | **+18.65%** | **+12.66%** |
| 32 | 1.741 ms/image | TinyCNN s48: 1.919 ms/image | 0.00 pp | **+10.26%** | **+7.23%** |

TinyCNN amortizes much better than handcrafted XGBoost: its measured routing
cost falls from about 0.69 ms/image at batch 8 to about 0.23--0.25 ms/image at
batch 32. It is still not cheap enough to beat p0 because p90 routes mainly to
p0 or p10 and the performance gap between those two engines is small.

For example, TinyCNN s45 selected:

| Batch | p0 batches/images | p10 batches/images | Other models |
|---:|---:|---:|---:|
| 8 | 147 / 1,176 | 103 / 824 | 0 |
| 16 | 97 / 1,552 | 28 / 448 | 0 |
| 32 | 55 / 1,744 | 8 / 256 | 0 |

The routed accuracy can be slightly higher than p0 because p0 and p10 make
different mistakes on individual images. This is subset-level model
complementarity, not evidence that a pruned model is globally more accurate.

## Interpretation and next step

The proposed batch-level design is valid and removes much of the per-image
TinyCNN overhead, but the **90th percentile is too conservative for batches 16
and 32 with the current score calibration**. It should remain a safety-oriented
reference. The next controlled comparison should keep the same images, engines,
batch sizes, and p0 baselines while sweeping the 50th, 70th, 80th, and 90th
percentiles. A useful policy must first satisfy an accuracy constraint and then
show positive measured savings after router overhead.

## Files

- `energy_baseline_comparison.csv`: all energy-policy means and p0-relative metrics;
- `latency_baseline_comparison.csv`: all latency-policy means and p0-relative metrics;
- `*_system_comparison.png`: accuracy, end-to-end latency, and VDD_IN energy;
- `*_model_selection_counts.png`: selected model counts for the 2,000 images.

