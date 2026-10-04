# Jetson CIFAR-100 TensorRT power tests

This folder contains the CIFAR-100 ResNet-18 deployment experiments run on the
Jetson Orin Nano: TensorRT evaluation, controlled power/clock sweeps, static
model measurements, and complete routed-system benchmarks.

```text
models/fp32/                 Local p0 and p10-p90 checkpoints; not stored in Git
manifests/models.sha256.csv  Model provenance and integrity hashes
manifests/power_profiles.*   Board-specific nvpmodel mappings
scripts/evaluate_trt.py      ONNX export, TensorRT build, evaluation and plots
scripts/run_power_sweep.py   Power-mode orchestration and combined results
scripts/benchmark_static_models.py  Fixed-image static batch/model sweep
scripts/benchmark_dynamic_routing.py  Complete router + selected-engine benchmark
scripts/benchmark_batch_routers.py  One-decision-per-batch router benchmark
scripts/build_batch_router_engine.py  TinyCNN ONNX/TensorRT export
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

Before routing, benchmark every candidate directly on the same fixed images:

```bash
python scripts/benchmark_static_models.py \
  --trt-engine-dir outputs/tensorrt_batch128/engines/fp32 \
  --output-dir results/static_models \
  --models fp32_p0,fp32_p30,fp32_p50,fp32_p70 \
  --batch-sizes 1,8,16,32,64,128 \
  --repeats 3
```

The static summary marks Pareto-efficient candidates using measured accuracy,
end-to-end latency, and VDD_IN energy.

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

The p0 baseline bypasses both the router and percentile calculation. It is a
direct, like-for-like p0 inference measurement at the same batch size.

Build TensorRT engines under the GPU clock profile used by the measurement.
On this Jetson, deserializing the 624.75 MHz plans while fixed at 612 MHz
triggered TensorRT's cross-device-profile warning. Separate batch-128 engine
directories were therefore built for the energy and latency profiles.

Energy and latency remain separate experiments. Copy selected small router
artifacts to an ignored directory, create a manifest from
`manifests/dynamic_routers.example.json`, and run:

```bash
python scripts/benchmark_dynamic_routing.py \
  --manifest manifests/dynamic_routers.energy.json \
  --trt-engine-dir outputs/tensorrt_batch128/engines/fp32 \
  --output-dir results/dynamic_routing_energy \
  --batch-sizes 8,16,32,64,128 \
  --difficulty-percentiles 50,70,80,90 \
  --candidate-models fp32_p0,fp32_p30,fp32_p50,fp32_p70 \
  --repeats 3
```

The p0 comparison is measured once for every batch size and reused for each
percentile, so all percentile policies are compared with the identical p0
observations. The current follow-up engines use a TensorRT optimization profile
with maximum batch 128.

## Low-overhead batch routers

The next-stage routers choose once per batch and use `p0`, `p10`, `p20`, or
`p30`. The tested router types are a one-feature decision stump, depth-3 tree,
logistic model, depth-2 XGBoost model, PyTorch batch TinyCNN, and TensorRT FP16
batch TinyCNN. The TinyCNN input is a 4x4 grid of 8x8 thumbnails sampled from
the batch.

Build a TensorRT router engine on the clock profile used by the benchmark:

```bash
python scripts/build_batch_router_engine.py \
  --checkpoint artifacts/routers/batch_routers/batch_tinycnn_s48.pth \
  --onnx artifacts/routers/batch_routers/batch_tinycnn_s48.onnx \
  --engine artifacts/routers/batch_routers/batch_tinycnn_s48_fp16.engine \
  --precision fp16
```

Then run the measured accuracy-budget sweep:

```bash
python scripts/benchmark_batch_routers.py \
  --manifest artifacts/routers/batch_routers/manifest_energy.json \
  --trt-engine-dir outputs/tensorrt_batch128/engines/fp32 \
  --output-dir results/batch_routers_energy \
  --batch-sizes 32,64,128 \
  --budgets-pp 0.5,1.0,1.5,2.0 \
  --repeats 3
```

Run energy and latency separately with engines built under their respective GPU
clock profiles.

Each completed measurement is written to the runs CSV immediately. Repeating
the same command resumes missing system, batch-size, budget, and repeat
combinations. Use `--no-resume` only for an intentional clean sweep.

Do not commit checkpoints, TensorRT engines, trained router artifacts, or raw
bulk outputs. Only source, concise CSV summaries, plots, and documentation
belong in Git.

## Reports

The 2026-09-30 hardware sweep, pruning and precision measurements, separate
energy/latency routing tests, plots, and notes are in
[`reports/jetson_2026_09_30`](reports/jetson_2026_09_30/README.md).

The 2026-10-01 follow-up with batch sizes 8/16/32 and 90th-percentile batch
difficulty is in
[`reports/jetson_batch90_2026_10_01`](reports/jetson_batch90_2026_10_01/README.md).

The expanded follow-up with rebuilt batch-128 engines, static Pareto baselines,
batches 8/16/32/64/128, difficulty percentiles 50/70/80/90, and separate energy
and latency policies is in
[`reports/jetson_batch_sweep_2026_10_01`](reports/jetson_batch_sweep_2026_10_01/README.md).

The 2026-10-03 comparison of low-overhead batch routers, `p10`/`p20` models,
separate energy and latency policies, model-selection counts, and static-model
baselines is in
[`reports/jetson_batch_routers_2026_10_03`](reports/jetson_batch_routers_2026_10_03/README.md).
