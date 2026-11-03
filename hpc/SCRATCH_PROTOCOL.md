# Scratch CIFAR classifier search, version 1

Generic experiment specification only. No cluster account, hostnames, connection
instructions or private storage paths are included. The removed HPC README is
not restored. Historical PC code, results and the pretrained pilot are unchanged.

## Scope

First optimize **ResNet-18 only**, at native 32x32 resolution. The torchvision
ResNet-18 receives a 3x3 stride-1/padding-1 initial convolution and no initial
max-pooling. This is explicitly `torchvision_resnet18_cifar_stem_v1`, not the
historical unmodified ImageNet-style 128x128 model. All weights are randomly
initialized, including the stem and 100-class output. No pretrained downloads.

The cached dataset must already exist. `prepare` uses a CPU allocation to verify
the official training dataset, reuse/verify the four-way split, hash its pixel
array, and freeze `plan.json` plus `data_ready.json`. A missing dataset fails
closed instead of occupying a GPU while downloading. Official test files are
present in the distributed archive but the final-test dataset is never constructed
or evaluated. Only classifier_train and classifier_validation have data loaders.

Roles remain 40,000 classifier training / 5,000 classifier validation / 3,000
router training / 2,000 router calibration. Fixed split seed 42 is independent of
training seeds. Non-overlapping indices do not guarantee duplicate-free photos.
QAT and INT8 calibration later use classifier-training data, not router data.

## Search (12 recipes x 200 epochs, seed 42)

SGD: learning rates .03/.1/.2, momentum .9, weight decay .0005.
Adam: learning rates .0003/.001/.003, betas .9/.999, weight decay .0001.
AdamW: learning rates .0003/.001/.003, betas .9/.999, weight decay .01.
These nine candidates use cosine decay over the full 200 epochs.

Candidate 9 is the published residual-network reference: SGD .1, momentum .9,
weight decay .0001, learning rate x.1 after epochs 80 and 120. Candidates 10/11
add CutMix alpha=1, probability=.5 to the SGD .1 and AdamW .001 cosine recipes.
Label weights use actual clipped patch area; validation never uses CutMix.

All use batch 128, four workers, crop(32,padding4), horizontal flip, fixed CIFAR
mean/std, no label smoothing, FP16 AMP, deterministic algorithms and no TF32.
Each seed starts from the same initial state across trials. Epoch-specific
worker seeds and newly constructed workers support epoch-boundary resumption.
Cross-device bitwise equality is NOT promised. Adam's coupled L2 and AdamW's
decoupled weight decay are distinct: this compares candidate recipes, not a
causal claim that one optimizer is intrinsically superior. This finite space is
not a global optimum, and its settings are not automatically optimal for the
seven other architectures. No new optimizer is included solely for novelty.

No early stopping/pruning based on short-run ranking: slow-starting recipes get
the full budget. Each trial's score is its best validation accuracy; validation
loss and ID break ties. Rank only after ALL 12 trials complete, verify their
checkpoint hashes, then freeze top two IDs in selection.json. Confirm each on
seeds45/48, full 200 epochs, and select by their mean best validation accuracy,
not the best seed. Preserve seed42 search results and both confirmation seeds.
Maximum: 2,400 search epochs plus 800 confirmation epochs = **3,200 epochs**.
Default array concurrency is two single-GPU jobs; it is a cap, not an allocation
request for two GPUs inside each single-GPU training process.

## Lifecycle and tracking

Run commands from repository root. Entry point: `python hpc/train_scratch_sweep.py
--help`. Actions: prepare, train, rank, finalize. Site-specific Slurm partition
and Python path are supplied at submission; the scripts do not choose a cluster.

1. CPU prepare job: `scratch_prepare.sbatch`, default STUDY_ACTION=prepare.
2. GPU preflight: `scratch_search.sbatch`, override array to 0 and set
   EPOCHS_THIS_JOB=3. These are the first three epochs of the SAME 200-epoch
   schedule, not an independently compressed three-epoch schedule.
3. Inspect preflight and continue trial0 with STUDY_RESUME=1; launch trials1..11
   without resume. Never overlap two workers for the same trial/seed directory.
4. CPU ranking: scratch_prepare.sbatch with STUDY_ACTION=rank. Writes
   leaderboard.csv and frozen selection.json; refuses partial search results.
5. GPU confirmations: scratch_search.sbatch with STUDY_PHASE=confirm and array
   0..3. Default search array0..11 must be overridden for confirmation.
6. CPU final selection: scratch_prepare.sbatch with STUDY_ACTION=finalize.

Per-batch progress, epoch metrics and checkpoint acknowledgements are printed
immediately to Slurm logs. Each trial directory has run.json, latest.pt (current
model/optimizer/scaler/scheduler AND best state), and summary.json with complete
epoch history, time, memory and settings. No W&B account, remote upload or extra
tracking dependency. The CSV/JSON records provide the initial offline tracker.
No training-energy or whole-system measurement is claimed.

Reruns require explicit --resume, exact source hashes, settings, library
versions, initial-state hash and split. Signals request a stop after a completed
epoch. A scheduler hard kill may lose the current epoch, not the last atomic
checkpoint. Crashes can leave active.lock: inspect the recorded job/host/PID
before any human-approved removal; never automatically delete a live lock.

## Literature anchors, not guarantees

- Bouniot et al., *Tailoring Mixup to Data for Calibration*, ICLR 2025,
  Appendix H: SGD .1, momentum .9, WD .0001, batch128, 200 epochs, steps80/120.
  https://proceedings.iclr.cc/paper_files/paper/2025/file/1eac91b56dc20d4f2046951453dde527-Paper-Conference.pdf
- *OpenMixup*, CIFAR benchmarks: CIFAR-adapted ResNets, multiple augmentation
  recipes and 200/400/800/1200 epoch budgets, not a universal best configuration.
  https://github.com/Westlake-AI/openmixup/blob/main/docs/en/mixup_benchmarks/Mixup_cifar.md
- SGDF optimizer comparison submission, Appendix D, Table5 (under-review status
  is not presented as acceptance): Adam/AdamW .001 and SGD .1 are baseline anchors.
  The candidate learning-rate neighborhoods and CutMix pairings are OUR design.
  https://openreview.net/pdf?id=fWNHKHh0Yn

The 5,000 validation images are deliberately used for HPO, so selection can
overfit validation. The untouched official test set will be used only after
settings/candidate policies are frozen. Do not promise literature accuracy from
our smaller classifier-training partition. Further data, scratch-vs-transfer,
resolution, architecture and augmentation changes require versioned protocols.
