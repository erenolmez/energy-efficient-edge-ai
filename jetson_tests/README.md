# Jetson CIFAR-100 TensorRT power tests

This folder contains only the material needed to reproduce the emailed
CIFAR-100 ResNet-18 results and repeat them under controlled Jetson power modes.

```text
models/fp32/                 Local p0 and p10-p90 checkpoints; not stored in Git
manifests/models.sha256.csv  Model provenance and integrity hashes
manifests/power_profiles.*   Board-specific nvpmodel mappings
scripts/evaluate_trt.py      ONNX export, TensorRT build, evaluation and plots
scripts/run_power_sweep.py   Power-mode orchestration and combined results
scripts/benchmark_dynamic_routing.py  Complete router + selected-engine benchmark
outputs/tensorrt/            Generated ONNX/engines; ignored by Git
results/power_sweep/         Generated CSVs and plots; raw outputs ignored
```

## Jetson environment

The copied environment was created at `/home/project/Desktop/resnet_env` using
Python 3.10.12. It includes NVIDIA PyTorch 2.4 (`nv24.7`), torchvision 0.20,
Torch-Pruning 1.3.5, pandas, matplotlib and seaborn. TensorRT is supplied by the
Jetson system installation because the environment enables system site packages.

The checkpoints are intentionally excluded from Git because the set is about
163 MB. `manifests/models.sha256.csv` identifies the exact required files. On
the current Jetson they already exist under the original `brand_new_pearl`
workspace and can be linked or copied into `jetson_tests/models/fp32`.

```bash
source /home/project/Desktop/resnet_env/bin/activate
cd /home/project/Desktop/energy-efficient-edge-ai/jetson_tests
python scripts/evaluate_trt.py --help
```

## Power sweep preparation

First inspect the modes actually supported by the board:

```bash
sudo nvpmodel -q --verbose
```

Create `manifests/power_profiles.jetson.json` from the example using only those
reported IDs. An nvpmodel profile is a total-module power configuration—not an
exact GPU-only watt limit. This Orin Nano exposes official 7 W and 15 W modes
plus GPU clocks of 306, 408, 510, 612 and 624.75 MHz. The supplied Orin Nano
manifest combines those supported settings. The experiment records both total
`VDD_IN` and combined `VDD_CPU_GPU_CV` rail power from `tegrastats`, then
calculates end-to-end energy as:

```text
energy (mJ/image) = measured power (W) × end-to-end latency (ms/image)
```

Preview without changing the board or running inference:

```bash
python scripts/run_power_sweep.py --dry-run
```

Run only after reviewing the profile mapping:

```bash
python scripts/run_power_sweep.py \
  --experiment-name hardware_characterization \
  --model-pattern '^distilled_resnet18$' \
  --precisions fp16 --batch-sizes 1,32
```

The default evaluation uses the complete CIFAR-100 test set as five fixed,
non-overlapping subsets of 2,000 images. The sweep restores the original
nvpmodel mode when it finishes or fails.

Some mode changes require a reboot on Orin Nano. In that case run the modes in
separate phases with `--mode-ids 1` and `--mode-ids 0`; completed profile CSVs
are automatically recombined. TensorRT engines are isolated per clock profile.

## Dynamic routing on Jetson

`benchmark_dynamic_routing.py` evaluates the exact same fixed 2,000 CIFAR-100
images used by the PC router evaluation. It can route individual images or make
one decision for a complete batch. For batch routing, it scores every image,
aggregates the scores with a configurable difficulty percentile, and runs the
entire batch through the selected TensorRT FP32 engine. It measures the complete
path: input-only router, routing decision, preprocessing, selected engine, and
synchronization. It also runs p0 with the same batch size and reports:

- p0-relative accuracy;
- total latency and throughput;
- measured VDD_IN and CPU/GPU/CV power;
- integrated energy per image;
- router overhead; and
- p0/p10/.../p90 selection counts for both images and batches.

Energy and latency remain separate experiments. Copy selected small router
artifacts to an ignored directory, create a manifest from
`manifests/dynamic_routers.example.json`, and run:

```bash
python scripts/benchmark_dynamic_routing.py \
  --manifest manifests/dynamic_routers.energy.json \
  --trt-engine-dir outputs/tensorrt/engines/fp32 \
  --output-dir results/dynamic_routing_energy \
  --batch-sizes 8,16,32 \
  --difficulty-percentile 90
```

The p0 comparison is recomputed separately for every batch size. Existing
engines in this project support batches up to 32; larger batches require
rebuilding all engines with a larger optimization profile.

Do not commit checkpoints, TensorRT engines, trained router artifacts, or raw
bulk outputs. Only source, concise CSV summaries, plots, and documentation
belong in Git.

## Latest measured results

The complete 2026-09-30 hardware sweep, mail-like reproduction, separate
energy/latency routing tests, plots, and discussion are in
[`reports/jetson_2026_09_30`](reports/jetson_2026_09_30/README.md).

The 2026-10-01 follow-up with batch sizes 8/16/32 and 90th-percentile batch
difficulty is in
[`reports/jetson_batch90_2026_10_01`](reports/jetson_batch90_2026_10_01/README.md).
