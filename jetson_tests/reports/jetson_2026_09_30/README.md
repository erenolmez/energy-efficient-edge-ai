# Jetson power and dynamic-routing results — 2026-09-30

## Scope

These experiments were run on the user's Jetson Orin Nano 8 GB using the existing CIFAR-100 distilled ResNet-18 family. Power is the Jetson `VDD_IN` module rail reported by `tegrastats`, and energy is integrated over the measured inference interval and divided by the number of images.

The work has three parts:

1. characterize one representative TensorRT FP16 p0 model across native `nvpmodel` modes and supported GPU clocks;
2. repeat the original mail-style FP32/FP16/INT8 pruning sweep with power, energy, and temperature added;
3. run two separate dynamic-routing experiments, one calibrated for energy and one for latency, against the normal unpruned FP32 p0 baseline.

## 1. Hardware characterization

The board exposes native 15 W and 7 W modes. The sweep used 306, 408, 510, 612, and 624.75 MHz where supported, at batch sizes 1 and 32.

| Operating point | Batch | E2E latency | VDD_IN | Energy/image | Accuracy |
|---|---:|---:|---:|---:|---:|
| 15 W / 306 MHz | 1 | 4.673 ms | 7.262 W | 33.930 mJ | 73.75% |
| 7 W / 306 MHz | 1 | 5.300 ms | 6.543 W | 34.685 mJ | 73.65% |
| 7 W / 408 MHz | 1 | 4.401 ms | 6.766 W | 29.785 mJ | 73.69% |
| 15 W / 612 MHz | 1 | **2.857 ms** | 8.961 W | 25.597 mJ | 73.75% |
| 15 W / 624.75 MHz | 1 | 2.875 ms | 8.616 W | **24.765 mJ** | 73.70% |

For single-image inference, 612 MHz produced the lowest latency and 624.75 MHz the lowest energy. The lowest-power mode did not minimize energy because the longer execution time offset its lower watts. This is a clear race-to-idle result and supports using joules per image rather than watts alone.

At batch 32, 15 W / 612 MHz was again fastest (0.553 ms/image), while 15 W / 408 MHz used the least energy (5.257 mJ/image).

## 2. Mail-like pruning and precision sweep

The accuracy results reproduce the earlier email values:

| Pruning | FP32 | FP16 | INT8 |
|---:|---:|---:|---:|
| 0% | 73.70% | 73.70% | 73.28% |
| 30% | 71.51% | 71.48% | 71.27% |

Therefore, the earlier interpretation still holds: most accuracy loss comes from pruning, while calibrated FP16/INT8 changes accuracy only slightly for this model family.

The newly controlled 624.75 MHz run does not exactly reproduce the old performance numbers. For unpruned p0, pure TensorRT throughput was 837, 4,108, and 5,993 images/s for FP32, FP16, and INT8. VDD_IN power was 10.82, 9.52, and 8.33 W. The accuracy and engine sizes agree with the original experiment, but throughput and power should not be compared directly with the old email unless GPU clock, `nvpmodel`, TensorRT build, batch, warm-up, and measurement protocol are identical.

## 3. Dynamic routing

Both routing tests used exactly the same fixed 2,000 CIFAR-100 test images for every system. Each full system was measured three times. The reported cost includes input analysis, router inference, selected-model inference, preprocessing, synchronization, and power integration. No oracle is used in the comparison.

### Energy objective — 15 W / 624.75 MHz

| System | Accuracy | Delta vs p0 | Latency | Energy | Energy saving vs p0 |
|---|---:|---:|---:|---:|---:|
| FP32 p0 baseline | 72.75% | 0.00 pp | 10.442 ms | 85.999 mJ | 0.00% |
| Compact XGBoost s45 | 72.50% | -0.25 pp | 13.034 ms | 96.502 mJ | -12.21% |
| **Compact XGBoost s48** | **72.50%** | **-0.25 pp** | **12.958 ms** | **95.616 mJ** | **-11.18%** |
| Compact XGBoost s51 | 72.40% | -0.35 pp | 13.030 ms | 96.683 mJ | -12.42% |
| Texture XGBoost s45 | 71.80% | -0.95 pp | 13.693 ms | 100.389 mJ | -16.73% |
| Texture XGBoost s48 | 72.05% | -0.70 pp | 13.513 ms | 98.196 mJ | -14.18% |
| Texture XGBoost s51 | 72.25% | -0.50 pp | 13.694 ms | 100.392 mJ | -16.74% |

Negative saving means the system consumed more than p0. The best energy router selected p0 511 times, p10 1,338 times, and p20 151 times. Its accuracy stayed close to p0, but the roughly 2.63 ms/image router overhead exceeded the savings from mostly choosing only mildly pruned models.

### Latency objective — 15 W / 612 MHz

| System | Accuracy | Delta vs p0 | Latency | Latency saving vs p0 | Energy change vs p0 |
|---|---:|---:|---:|---:|---:|
| FP32 p0 baseline | 72.75% | 0.00 pp | 10.569 ms | 0.00% | 0.00% |
| Compact XGBoost s45 | 72.15% | -0.60 pp | 13.076 ms | -23.72% | -10.79% |
| Compact XGBoost s48 | 72.40% | -0.35 pp | 13.125 ms | -24.19% | -11.82% |
| Compact XGBoost s51 | 72.15% | -0.60 pp | 13.144 ms | -24.37% | -11.75% |
| **TinyCNN-8 s45** | **72.20%** | **-0.55 pp** | **12.389 ms** | **-17.23%** | **-5.69%** |
| TinyCNN-8 s48 | 72.50% | -0.25 pp | 12.556 ms | -18.80% | -9.19% |
| TinyCNN-8 s51 | 72.25% | -0.50 pp | 12.512 ms | -18.39% | -7.27% |

The fastest routed system was TinyCNN-8 seed 45. It selected p0 394 times, p10 971 times, p20 574 times, p50 59 times, and p60 twice. It still lost to p0 because the measured routing path cost about 2.70 ms/image.

## Interpretation

The earlier PC-side estimates showed potential model-only savings, but the complete Jetson system does not yet save latency or energy. This is not evidence that input-based routing is impossible. It shows that the present Python/per-image implementation has too much overhead and that the accuracy-constrained policies choose p10/p20 too often, whose measured cost gap from p0 is too small.

The next technically justified steps are:

1. export the router to TensorRT or implement handcrafted features in optimized C++/CUDA;
2. share preprocessing between router and selected model instead of converting/resizing the same image twice;
3. train/calibrate policies with measured Jetson full-system costs, including router overhead;
4. test batched routing for throughput/energy, where router overhead can be amortized;
5. use a smaller set of better-separated candidates such as p0, p30/p40, and p70/p80;
6. rebuild the selected deployment engines on the current Jetson configuration before publication-quality performance reporting.

TensorRT emitted a warning when loading some legacy engines about engine plans built for a different device model. Accuracy remained consistent, but absolute routing and mail-like performance should be confirmed after rebuilding the final selected engines.

## Files

- `hardware/`: power-sweep summary and plots.
- `mail_like/`: precision/pruning summary and mail-style plots with energy and temperature.
- `routing/`: raw repeat-level routing results, p0-relative tables, model-selection plots, and system comparison plots.
