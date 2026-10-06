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
