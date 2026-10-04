# Energy-Efficient Edge AI

This repository contains the experiments for an MSc thesis on input-aware
inference on NVIDIA Jetson Orin Nano. The current study asks whether a small
router can select a pre-trained model before inference and reduce measured
energy or latency without exceeding an accuracy limit.

The experiments use CIFAR-100 and a distilled ResNet-18 family with structured
pruning. Energy and latency are treated as separate objectives. Deployment
claims use complete Jetson measurements, including input analysis, routing,
transfers, synchronization, and selected-model inference.

## Current result

Under a maximum accuracy loss of 0.5 percentage points, static `p10` is the
best tested deployment. Across batches 32, 64, and 128 it saves 8.80--10.13%
energy and 7.16--7.98% latency relative to the unpruned FP32 `p0` baseline.
The tested dynamic routers do not recover their own overhead at this limit.

Dynamic routing becomes competitive under looser constraints. Its clearest
advantage is a small energy improvement over the best eligible static model at
a 1.5-point allowance for batches 64 and 128. See the
[latest Jetson report](jetson_tests/reports/jetson_batch_routers_2026_10_03/README.md)
for the complete tables and figures.

## Repository layout

```text
architecture_tests/resnet_pruning/  ResNet-34/50 structural pruning check
pc_tests/router_benchmark/           Router training and PC-side comparison
jetson_tests/                        TensorRT, power, static and routed benchmarks
progress_reports/                    Dated LaTeX reports and compiled PDFs
tests/                               Environment-independent unit tests
```

All runtime routers receive only the input image. Candidate predictions and
measured Jetson costs are used offline for training and policy calibration;
candidate classifiers are not run before the routing decision.

## Main experiment

The current batch study uses:

- FP32 `p0`, `p10`, `p20`, and `p30` inference models;
- a gradient stump, decision tree, logistic regression, XGBoost, PyTorch
  TinyCNN, and FP16 TensorRT TinyCNN;
- batches 32, 64, and 128;
- accuracy allowances of 0.5, 1.0, 1.5, and 2.0 percentage points;
- separate energy and latency calibration;
- the same fixed, stratified 2,000 CIFAR-100 images for every system.

The Jetson instructions are in [jetson_tests/README.md](jetson_tests/README.md).
The PC training protocol is in
[pc_tests/router_benchmark/README.md](pc_tests/router_benchmark/README.md).

## Progress reports

The advisor-facing reports are indexed in
[progress_reports/README.md](progress_reports/README.md). Each report includes
the LaTeX source, required figures, and the PDF compiled from that source.

## Data and generated files

Model checkpoints, virtual environments, TensorRT engines, trained router
artifacts, and large raw outputs are excluded from Git. Manifests and concise
CSV summaries identify the inputs used for the tracked results.
