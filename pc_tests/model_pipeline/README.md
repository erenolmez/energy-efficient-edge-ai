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

TensorRT FP32, FP16 and INT8 engines are built from the same validated FP32
ONNX graph. FP16 is selected by the TensorRT builder, while INT8 additionally
uses a fixed calibration subset. Creating renamed FP16 or INT8 checkpoint
copies would not perform quantization and is therefore avoided.

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
