# Larger router-training pool audit — 12 October 2026

Actual manifest: artifacts/baselines/cifar100_split_seed42.json.
File SHA256: 306cc12e260a94ff157948cd583d7830e67e2274e8e6fbf539cc85b8a957c88f.
Split identity: 413f355e2f5eb0184de1355b63f6a3743cd1f5cdb3d7cc682e3fbc89d37252aa.

Verified 45,000 unique classifier-training indices and 5,000 unique validation
indices; overlap zero, union all50,000 original CIFAR-100 training images.
Unused original training images: zero. All eight baseline summary records carry
the same split identity and smoke_test=false. Dataset construction in
model_pipeline/data.py uses these training indices for classifier fitting.

Therefore a larger reserved pool unseen by the existing classifiers does not
exist in this split. No held-out test samples may be moved into router fitting.
The existing 5,000 validation images are the only reserved original-training
partition in this manifest (they also informed baseline checkpoint selection).

Two options require a user choice before changing the data protocol:

1. Faster exploratory study: train routers on candidate correctness from the
   45,000 classifier-training images, keep tuning/calibration separate from
   router fitting and retain held-out tests solely for evaluation. Declare
   in-sample classifier supervision and possible optimism/distribution gap;
   this is not out-of-sample correctness supervision. Generating213 predictions
   per training input is new GPU work, not a rerun of valid static measurements.
2. Cleaner out-of-sample study: retrain classifiers with a larger reserved
   router-training partition or generate out-of-fold candidate predictions.
   This requires new classifier training and deployment/profiling artifacts;
   preserve existing213 results as controls. Substantially greater runtime.

Neither option was launched by this audit. Current120-case small-data baseline
is complete and preserved. No scalar/legacy or Jetson work resumed. Current
GPU temperature62C at the audit check; old guard70C is historical, not live.
