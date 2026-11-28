#!/bin/bash
#SBATCH --job-name=a40-newrouters
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_a40
#SBATCH --exclude=compute-5-11,compute-7-13
# Job 3, router comparison on the A40 (gpu_a40), all in one job; test set never read.
#   1. cross-fit training of every router family ON THE A40 GPU (5 folds on the 5k router pool,
#      no classifier-validation images in the main routers): direct-tiny16, direct-wide32,
#      blended-tiny16, lookup-res32, plus the new batch-set, smallest-sufficient (ordinal) and
#      disagreement routers. Existing families carry their A40 TensorRT costs from a40_routers_v1.
#   2. A40 TensorRT FP32 engines + measured cost for the three new routers (hpc.a40_newrouter_costs).
#   3. Policy locks, random mix and report on the measured A40 numbers, for shuffled batches and for
#      class-coherent batches (4 classes per batch). Step 3 is numpy re-analysis of the A40 numbers.
# Submit with absolute --chdir and --output (see the command in the thread).
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through the scheduler}"
SCRATCH_FLASH="${SCRATCH_FLASH:-/mnt/beegfs/eolmez}"
STUDY_BASE="${STUDY_BASE:-${SCRATCH_FLASH}/clean_classifier_study}"
CIFAR_DATA="${CIFAR_DATA:-${SCRATCH_FLASH}/datasets/cifar100}"
SCREEN="${SCREEN:-${STUDY_BASE}/router_arch_screen_v1}"
STAGE_A="${STAGE_A:-${STUDY_BASE}/a40_engines_v1}"
ROUTER_COSTS="${ROUTER_COSTS:-${STUDY_BASE}/a40_routers_v1/router_costs.json}"
OUT="${CROSSFIT_OUT:-${STUDY_BASE}/a40_crossfit_v2}"
REPO="$(cd "${SLURM_SUBMIT_DIR:?}" && pwd)"
cd "$REPO"
test -f hpc/a40_crossfit.py && test -f hpc/a40_newrouter_costs.py
source hpc/a40_engines_env.sh   # overlays + import preflight; edge-ai env untouched
test -f "$ROUTER_COSTS" && test -f "$STAGE_A/predictions.npz" && test -f "$STAGE_A/engines.json"
test ! -e "$OUT" || { echo "$OUT exists; set CROSSFIT_OUT to a new directory" >&2; exit 1; }
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
hostname; date -u; git rev-parse HEAD; nvidia-smi --query-gpu=name,uuid --format=csv
"$HPC_PYTHON" -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print('GPU', torch.cuda.get_device_name(0))"
echo "== 1/3 cross-fit training on the GPU"
srun --ntasks=1 "$HPC_PYTHON" -m hpc.a40_crossfit --phase train --base "$STUDY_BASE" --data "$CIFAR_DATA" \
    --screen "$SCREEN" --stage-a "$STAGE_A" --output "$OUT" --router-costs "$ROUTER_COSTS"
echo "== 2/3 A40 TensorRT cost of the new routers"
srun --ntasks=1 "$HPC_PYTHON" -m hpc.a40_newrouter_costs --crossfit "$OUT" --stage-a "$STAGE_A" \
    --base "$STUDY_BASE" --data "$CIFAR_DATA"
echo "== 3/3 locks and reports (numpy re-analysis of the A40 numbers): shuffled, then 4 classes per batch"
bash "$REPO/hpc/a40_crossfit_relock.sh" "$OUT"
bash "$REPO/hpc/a40_crossfit_relock.sh" "$OUT" 4
bash "$REPO/hpc/a40_crossfit_relock.sh" "$OUT" 4 fold
echo "Reports: $OUT/crossfit_report.md, $OUT/coherent4/crossfit_report.md, $OUT/coherent4_fold/crossfit_report.md"
date -u
