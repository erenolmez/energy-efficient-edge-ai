# Classifier pipeline

This directory contains the shared CIFAR-100 baseline-training entry point for
the architecture study. The selected models are ResNet-18/34/50/101,
MobileNetV3-Small/Large, EfficientNet-B0 and ShuffleNetV2 1.0x.

All models use the same normalization, 128x128 input, standard CIFAR-100 test
set and a seed-42 stratified 45,000/5,000 training/validation split. The split
manifest stores the exact indices and a SHA-256 fingerprint.

Check construction before starting a long run:

```powershell
python inspect_architectures.py --device cuda --output artifacts/architecture_check.json
```

Train the unpruned FP32 baselines:

```powershell
python train_baselines.py `
  --data-dir "C:\Users\Eren\Desktop\tez\pruning + quantization\data" `
  --output-dir artifacts\baselines `
  --device cuda `
  --amp `
  --resume
```

`--smoke-batches 1 --epochs 1` checks the complete data and checkpoint path
without producing a scientifically meaningful result. Files produced by smoke
tests must not be used in comparisons.

Training writes one whole-model `.pth` file and a metadata `.json` sidecar for
each architecture. Generated checkpoints remain excluded from Git.

## Structured pruning

Pruning starts from each completed FP32 baseline. Every requested level is
created independently from `p0`, then recovery-trained on the same seed-42
split. The requested channel ratio and the measured parameter/MAC reductions
are both recorded because dependency-aware pruning does not make them equal.

Check that every architecture can be pruned and still produce 100 logits:

```powershell
python prune_finetune.py `
  --data-dir "C:\Users\Eren\Desktop\tez\pruning + quantization\data" `
  --structure-only --device cuda
```

Run the complete p10--p90 recovery experiment:

```powershell
python prune_finetune.py `
  --data-dir "C:\Users\Eren\Desktop\tez\pruning + quantization\data" `
  --baseline-dir artifacts\baselines `
  --output-dir artifacts\pruned `
  --epochs 20 --batch-size 128 --num-workers 4 `
  --device cuda --amp --resume
```

The job is resumable at each epoch and skips finished architecture/ratio
combinations. Quantization is intentionally a later stage so that FP16 and
INT8 variants are always derived from the recovered pruned FP32 models.

Regenerate the consolidated Task 3 summary, table and SVG figures from the
completed metadata sidecars without rerunning training:

```powershell
python summarize_pruning.py --input-dir artifacts\pruned
```

This writes `pruning_summary.json`, `pruning_summary.csv`,
`pruning_results.md`, and plots for test accuracy, measured MAC reduction,
measured parameter reduction and recovery fine-tuning time. The pruning entry
point also refreshes these files after every completed model, including when a
run is restricted to one architecture.

Torch-Pruning 1.6.1 cannot construct a dependency graph for torchvision's
ShuffleNetV2 channel-shuffle reshape. The reproducible pruning sweep therefore
excludes ShuffleNetV2 rather than applying a different pruning definition to
that one family. Its unpruned baseline remains available for later precision
and deployment comparisons.

## Precision-build preparation

Numerically compare every FP32 export with its PyTorch source on the same
fixed CIFAR-100 test images at batches 1, 8 and 128:

```powershell
python validate_exports.py `
  --manifest artifacts\quantization\deployment_manifest.json `
  --data-dir "C:\Users\Eren\Desktop\tez\pruning + quantization\data" `
  --output reports\onnx_parity.json
```

The report records input indices, input SHA-256, runtime versions, output
differences, top-1 agreement and dynamic-batch execution. These equivalence
checks do not replace a full test-set accuracy evaluation.

Later TensorRT engines can be built from the validated FP32 ONNX graphs.
The PC precision study below is separate: it creates actual half-precision
weights and calibrated INT8 QDQ graphs, not renamed FP32 copies.

Prepare the deployment manifest, deterministic calibration indices and a small
ONNX smoke test:

```powershell
python prepare_quantization.py `
  --baseline-dir artifacts\baselines `
  --pruned-dir artifacts\pruned `
  --output-dir artifacts\quantization `
  --smoke-models resnet18__p0__fp32,resnet18__p20__fp32
```

The complete export is a separate gated stage. Do not pass every model through
`--smoke-models` when only checking the pipeline.

Run the resumable full export after the smoke stage succeeds:

```powershell
python prepare_quantization.py `
  --baseline-dir artifacts\baselines `
  --pruned-dir artifacts\pruned `
  --output-dir artifacts\quantization `
  --export-all
```

Current, valid ONNX files are reused. `export_validation.json` is refreshed
after every model so interrupted runs can be audited and resumed.

Screen deployment candidates against each architecture's own p0 baseline:

```powershell
python screen_candidates.py `
  --deployment-manifest artifacts\quantization\deployment_manifest.json `
  --output-dir artifacts\quantization `
  --budgets 0.5,1.0,1.5,2.0
```

For each accuracy-loss budget, selection maximizes measured MAC reduction and
uses parameter reduction and accuracy as tie-breakers. The result is a compact
`jetson_candidates.json` manifest; precision engines are not built during this
screening step.

Prepare the final PC-side deployment bundle without connecting to the Jetson:

```powershell
python prepare_deployment_bundle.py `
  --candidate-manifest artifacts\quantization\jetson_candidates.json `
  --calibration-manifest artifacts\quantization\calibration_indices.json `
  --output-dir artifacts\quantization\jetson_bundle
```

The default uses hard links for ONNX models on the same filesystem, avoiding a
second physical copy of large model data. `bundle_manifest.json` records file
sizes and SHA-256 hashes. TensorRT engines must still be built on the target
Jetson and are deliberately absent from this PC-only bundle.

## PC precision conversion and accuracy

Evaluate every source, including those outside the provisional shortlist:

```powershell
python run_precision_experiments.py `
  --manifest artifacts\quantization\deployment_manifest.json `
  --calibration-manifest artifacts\quantization\calibration_indices.json `
  --data-dir "C:\Users\Eren\Desktop\tez\pruning + quantization\data" `
  --output-dir artifacts\pc_precision --report-dir reports
```

The run covers 71 sources in FP32, FP16 and INT8 (213 full-test evaluations).
FP16 checkpoints contain half-precision weights and have matching ONNX graphs.
INT8 uses training-only MinMax calibration on 1,024 fixed images, symmetric
signed activations and per-channel weights for Conv, Gemm and MatMul. Other
operators remain floating point. Calibration settings are fixed before test
evaluation; large accuracy losses are retained in the results.

FP32 and INT8 run in ONNX Runtime on CPU; FP16 runs in PyTorch on CUDA.
Evaluation duration must not be used as a precision speedup comparison across
these runtimes. These artifacts are not TensorRT engines.

Completed records include model hashes, accuracy on all 10,000 test images,
conversion details and saved predictions. Restarting the command verifies
hashes and skips completed records. Models, input caches and predictions stay
under ignored artifact directories; small result tables and figures are tracked.

Generate the final tables and plots after all evaluations finish:

```powershell
python summarize_precision.py --report-dir reports
```

The summary compares every result with the freshly evaluated FP32 p0 of its
own architecture and separates pruning loss from conversion loss.

## October 20 report code and measured controls

The four-page report, code-to-table map, small evidence files, publication checks
and replay limitations are documented in
[`progress_reports/2026-10-20/README.md`](../../progress_reports/2026-10-20/README.md).
Scalar/correctness router code, uniform-random comparisons, fixed p80/p90 controls
and their runtime helpers live here; test modules live in `tests/`. Publishing
these sources does not start or authorize another experiment. Large model,
engine, image and prediction artifacts remain local and ignored.
