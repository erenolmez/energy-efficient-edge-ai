# Jetson CIFAR-100 TensorRT power tests

This folder contains only the material needed to reproduce the emailed
CIFAR-100 ResNet-18 results and repeat them under controlled Jetson power modes.

```text
models/fp32/                 Local p0 and p10-p90 checkpoints; not stored in Git
manifests/models.sha256.csv  Model provenance and integrity hashes
manifests/power_profiles.*   Board-specific nvpmodel mappings
scripts/evaluate_trt.py      ONNX export, TensorRT build, evaluation and plots
scripts/run_power_sweep.py   Power-mode orchestration and combined results
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
cd /home/project/Desktop/brand_new_pearl/jetson_tests
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
python scripts/run_power_sweep.py
```

The default evaluation uses the complete CIFAR-100 test set as five fixed,
non-overlapping subsets of 2,000 images. The sweep restores the original
nvpmodel mode when it finishes or fails.
