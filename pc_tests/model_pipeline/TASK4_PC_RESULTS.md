# Task 4: PC-side deployment preparation

## Scope

This stage prepared trained CIFAR-100 models for later TensorRT evaluation. It
did not build TensorRT engines or measure Jetson performance.

## Export

- 71 FP32 source checkpoints were exported and validated as dynamic-batch ONNX
  graphs: eight unpruned baselines and 63 structured-pruning variants.
- The ONNX files occupy 1.70 GB in total.
- Each graph can later be used to build FP32, FP16 and calibrated INT8 TensorRT
  engines, giving 213 possible precision targets.
- INT8 calibration uses a fixed seed-42 subset of 1,024 CIFAR-100 training
  images. Its indices and checksum are stored with the deployment metadata.

## Numerical export verification

All 71 sources passed PyTorch-to-ONNX Runtime comparison at batches 1, 8 and
128: 213 execution checks. Every check produced finite outputs with the
expected batch-by-100 shape and passed elementwise comparison at absolute and
relative tolerances of 0.001. The maximum absolute logit difference was
0.000280; top-1 predictions agreed on every checked image.

The checks used the same seed-42 sample of 128 CIFAR-100 test images with the
training pipeline's evaluation transform. PyTorch ran FP32 on CUDA and ONNX
Runtime ran on the CPU; CUDA TF32 was disabled. The test took 67.31 seconds.
This confirms export equivalence on the sample, not full test-set accuracy.
The machine-readable record includes image indices, input hash and runtime
versions in [reports/onnx_parity.json](reports/onnx_parity.json).

The complete pruning table and four plots are in [reports](reports).
All 31 repository unit tests passed during PC finalization.

## Candidate screening

This is a provisional descriptive shortlist based on previously reported
test-set accuracy, not an independently validated model-selection procedure.
Deployment selection should use validation data and then evaluate its result
on a held-out test set. All 71 source models are preserved. MAC reductions are
compute-count reductions, not measured latency or energy savings, and this
shortlist does not restrict the model pool for future routing experiments.

Each architecture was compared with its own p0 test accuracy. Candidate models
were screened at maximum accuracy losses of 0.5, 1.0, 1.5 and 2.0 percentage
points. Within a budget, measured MAC reduction was maximized; parameter
reduction and accuracy were tie-breakers.

The screening retained nine unique sources (27 precision targets): the eight
p0 models and MobileNetV3-Small p10. MobileNetV3-Small p10 qualified only for
the 2.0-point budget:

- p0 test accuracy: 62.44%
- p10 test accuracy: 60.82%
- accuracy loss: 1.62 percentage points
- measured MAC reduction: 27.11%

No pruned ResNet, EfficientNet-B0 or MobileNetV3-Large model satisfied the
2.0-point limit. This means the present pruning-recovery settings do not yet
provide useful strict-accuracy candidates for those families.

## Local deployment bundle

`artifacts/quantization/jetson_bundle` contains the nine selected ONNX graphs,
candidate and calibration manifests, file sizes and SHA-256 hashes. The model
files are hard-linked to the validated export directory, so the bundle does
not consume a second physical copy of the ONNX data.

## Remaining work

FP16 and INT8 conversions have not been performed on the PC. The 213 entries
are planned precision targets, not 213 converted models. Calibration indices
have been prepared; calibration itself has not been run.

TensorRT FP32/FP16/INT8 engine building, post-conversion accuracy checks, and
latency, throughput, power, energy and temperature measurements remain Jetson
tasks. They should begin only after the PC-side bundle is transferred and its
checksums are verified on the target device.
