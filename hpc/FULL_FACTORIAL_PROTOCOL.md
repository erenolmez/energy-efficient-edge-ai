# Full factorial CIFAR-32 study

User-requested exhaustive evaluation of a finite, frozen grid. This replaces
the small candidate screen as the new main study; older jobs and checkpoints
are retained separately, not relabelled or overwritten.

- Eight original architecture families: ResNet18/34/50/101, MobileNetV3 small
  and large, EfficientNet-B0, ShuffleNetV2 x1.0.
- SGD rates .03/.1/.2, Adam and AdamW rates .0003/.001/.003.
- SGD/Adam weight decay .0001/.0005; AdamW .01/.05.
- Cosine200 or step80/120 (factor .1); ordinary augmentation or CutMix.
- Every combination trains seeds42/45/48 for 200 epochs.
- 1728 total runs, 345600 maximum epochs, 72 recipes per architecture
  (216 runs including seeds), no validation-selected early stopping.

This is a multi-day computation, potentially hundreds of GPU-hours or more.
Checkpoints may require hundreds of GB. Shared filesystem capacity does not
prove a user's quota; account access, allocations and queue time remain external.
The first allocation covers at most24 hours per worker, NOT a promise that the
entire study finishes. No huge job-array submission: arrays are disabled.

The same disjoint40k/5k/3k/2k manifest is reused. Only classifier training and
validation are loaded. No router subsets, official test evaluation, pruning,
quantization or external weights. Best validation checkpoint per run is retained;
final configuration comparisons must aggregate all three seeds, not pick a seed.

Torchvision implementations are used with weights=None. ResNets replace the
stem with3x3 stride1 and remove maxpool. Other families change only the first
convolution stride to1; ShuffleNet also removes initial maxpool. Later block
strides are unchanged. These adaptations are explicit research choices, not a
claim of a universally optimal published CIFAR recipe; implementations are not
identical to historical timm models at128px. All eight CPU forward/backward
shape/finite-gradient checks must pass before publication. GPU validation
comes from actual allocated training; no claim that CPU tests prove GPU parity.

submit_full_factorial.py --submit --workers4 submits CPU preparation and four
independent one-GPU jobs with afterok dependency. New jobs are in addition to
any pre-existing pilot/small-study jobs. It requires inherited private W&B
authentication and explicit project/entity, never keys in code. Worker claims
use exclusive directory creation. Each case has its own train.log, checkpoint,
summary and W&B run in one study group. Submission receipts block duplicates.
No existing files are deleted. The original trainer and its dependencies remain
unchanged so its already-running checkpoint stays valid.

Expected nonfinite-loss numerical failures are retained and remaining cases
continue. Unexpected errors stop the worker and request other workers stop
after their current case; failure is never retried blindly. Walltime signal
requests save-after-epoch and retains the paused claim for explicit verified
continuation. A machine crash may leave interrupted claims; do not delete them
or interpret them as currently active without checking scheduler identity.

full_factorial_status.py is read-only and prints per-case epoch percentages,
complete/failed/paused/unclaimed counts. It does not select winners from an
incomplete grid. Existing W&B privacy allowlists/local checkpoint authority apply.
