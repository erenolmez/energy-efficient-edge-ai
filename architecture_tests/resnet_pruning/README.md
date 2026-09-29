# ResNet-34/50 Torch-Pruning compatibility check

`validate_torch_pruning.py` is a structural smoke test. It verifies that
dependency-aware structured pruning can modify randomly initialized ResNet-34
and ResNet-50 models while retaining a 100-class output.
It records actual MAC and parameter reductions because the requested channel
ratio is not the same as final whole-model compression.

It uses a random 128x128 tensor, not CIFAR-100 images. It therefore does not
measure accuracy, fine-tuning recovery, TensorRT latency, or Jetson energy. A
valid CIFAR-100 comparison requires separately trained baselines followed by
pruning and equal-schedule fine-tuning.

## Structural results

All 18 combinations completed a forward pass with output shape `[1, 100]`.

| Requested ratio | ResNet-34 MAC reduction | ResNet-34 parameter reduction | ResNet-50 MAC reduction | ResNet-50 parameter reduction |
|---:|---:|---:|---:|---:|
| 10% | 22.59% | 21.78% | 21.84% | 20.89% |
| 20% | 40.14% | 37.80% | 38.96% | 37.17% |
| 30% | 53.81% | 52.76% | 52.99% | 52.15% |
| 40% | 67.10% | 65.00% | 66.05% | 64.60% |
| 50% | 74.12% | 74.91% | 74.08% | 74.72% |
| 60% | 84.91% | 85.15% | 84.56% | 84.61% |
| 70% | 92.06% | 91.60% | 91.62% | 91.26% |
| 80% | 96.55% | 96.46% | 96.32% | 96.21% |
| 90% | **77.06%** | 98.03% | **86.86%** | 98.47% |

Torch-Pruning therefore works structurally for both architectures. However,
the 90% setting is abnormal: parameter reduction continues to increase while
reported MAC reduction becomes worse than at 80%. This reproduces the kind of
non-monotonic high-pruning behavior seen in the ResNet-18 deployment results.
It is evidence that 90% should be treated as an unstable extreme configuration,
not as evidence that pruning entirely failed.

Raw machine-readable measurements are in `results.csv`.

Install `requirements.txt` into the CUDA environment before rerunning this
compatibility check. CUDA-enabled PyTorch is installed separately.

Recommended full accuracy experiment:

1. Train each baseline with the same image size, transforms, epochs, optimizer,
   and seed.
2. Evaluate the unpruned checkpoint on the shared CIFAR-100 test set.
3. Start with pruning ratios 0.3, 0.5, 0.7, and 0.9.
4. Fine-tune every pruned network using the same schedule.
5. Report requested ratio, actual MAC/parameter reduction, Top-1 accuracy,
   latency, and energy. Treat 0.9 as an extreme stress test, not an expected
   operating point.
