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

Torch-Pruning 1.6.1 cannot construct a dependency graph for torchvision's
ShuffleNetV2 channel-shuffle reshape. The reproducible pruning sweep therefore
excludes ShuffleNetV2 rather than applying a different pruning definition to
that one family. Its unpruned baseline remains available for later precision
and deployment comparisons.
