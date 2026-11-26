# Energy-Efficient Edge AI

This MSc thesis asks when a lightweight router can choose one classifier for a
CIFAR-100 batch and reduce end-to-end latency or GPU energy without exceeding
an accuracy-loss budget. Latency and energy are separate objectives. Classifier
training and development screening run on HPC; deployment measurements report
the specific PC or Jetson hardware and runtime used.

## Start here

- [Read the latest published report (24 November 2026, PDF)](progress_reports/2026-11-24/progress_report.pdf)
- [Find the experiment code and reproduction notes](progress_reports/2026-11-24/README.md)
- [See the historical distilled ResNet-18 Jetson results](jetson_tests/reports/jetson_batch_routers_2026_10_03/README.md)

## Latest published results

The [24 November report](progress_reports/2026-11-24/) covers the PC batch-routing
comparison against fixed ResNet101, the HPC scratch and ImageNet-initialized
classifier comparison, and structured pruning with per-class evaluation. It
reports 16 completed ImageNet-initialized evaluations (eight architectures at
32×32 and 128×128) and 63 pruning cases across seven scratch-trained architectures.
These are distinct from the earlier PC pruning and ONNX export bank documented in
the [PC architecture results](pc_tests/model_pipeline/TASK4_PC_RESULTS.md).
PC measurements and HPC classifier evaluations are not Jetson deployment results.

**Follow-up work (not in the published report):** a separate PC benchmark
compares six router variants on the frozen 128×128 HPC model bank. HPC
development screening compares router architectures and the data used to
estimate class profiles. These are separate from the planned A40 end-to-end
comparison; no A40 latency or energy results are claimed here.

## Historical: distilled ResNet-18 on Jetson

Under a maximum accuracy loss of 0.5 percentage points, static `p10` is the
best tested deployment. Across batches 32, 64, and 128 it saves 8.80--10.13%
energy and 7.16--7.98% latency relative to the unpruned FP32 `p0` baseline.
The tested dynamic routers do not recover their own overhead at this limit.

Dynamic routing becomes competitive under looser constraints. Its clearest
advantage is a small energy improvement over the best eligible static model at
a 1.5-point allowance for batches 64 and 128. See the
[Jetson batch-router report](jetson_tests/reports/jetson_batch_routers_2026_10_03/README.md)
for the complete tables and figures.

## Repository layout

| Path | What belongs there |
| --- | --- |
| [`model_pipeline/`](model_pipeline/) | **Shared model and data library:** architecture definitions, data, naming, and reproducibility. |
| [`pc_tests/model_pipeline/`](pc_tests/model_pipeline/README.md) | **PC experiments and GPU runtime:** [router implementations](pc_tests/model_pipeline/routers/), [TensorRT and measurement helpers](pc_tests/model_pipeline/runtime/), and [experiment runners](pc_tests/model_pipeline/experiments/). Root-level compatibility scripts keep old commands working. |
| [`hpc/`](hpc/) | **Slurm training and evaluation:** [scratch classifier sweep protocol](hpc/FULL_FACTORIAL_PROTOCOL.md), [ImageNet comparison trainer](hpc/train_imagenet_comparison.py), held-out evaluation, and pruning. |
| [`jetson_tests/`](jetson_tests/README.md) | **Jetson hardware experiments:** TensorRT inference, controlled power and clock settings, and static and routed measurements. |
| [`pc_tests/router_benchmark/`](pc_tests/router_benchmark/README.md) | **Historical PC router benchmark:** the distilled FP32 ResNet-18 pruning family, not the newer eight-architecture bank. |
| [`architecture_tests/resnet_pruning/`](architecture_tests/resnet_pruning/) | **Pruning feasibility checks** for ResNet-34/50; not the main experiment or the software test suite. |
| [`tests/`](tests/) | **Software tests:** policy, data, protocol, runtime helper, and provenance checks. |
| [`progress_reports/`](progress_reports/README.md) | **Advisor-facing reports:** dated PDFs, LaTeX sources, figures, and evidence links. |

## Progress reports

The advisor-facing reports are indexed in
[progress_reports/README.md](progress_reports/README.md). Each report includes
the LaTeX source, required figures, and the PDF compiled from that source.

## Data and generated files

Model checkpoints, virtual environments, TensorRT engines, trained router
artifacts, and large raw outputs are excluded from Git. Manifests and concise
CSV summaries identify the inputs used for the tracked results.
