#!/bin/bash
#SBATCH --job-name=a40-routers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_a40
#SBATCH --account=eolmez
#SBATCH --qos=normal
#SBATCH --exclude=compute-5-11,compute-7-13
#SBATCH --output=a40-routers-%j.log
# Job 3, stage B1: router calibration scores, TensorRT router engines and A40 router costs,
# then the stage B2 policy lock (CPU). Needs stage A (a40_engines_v1). Never opens the test set.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through the scheduler}"
SCRATCH_FLASH="${SCRATCH_FLASH:-/mnt/beegfs/eolmez}"
STUDY_BASE="${STUDY_BASE:-${SCRATCH_FLASH}/clean_classifier_study}"
CIFAR_DATA="${CIFAR_DATA:-${SCRATCH_FLASH}/datasets/cifar100}"
SCREEN="${SCREEN:-${STUDY_BASE}/router_arch_screen_v1}"
STAGE_A="${STAGE_A:-${STUDY_BASE}/a40_engines_v1}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f hpc/a40_routers.py
source hpc/a40_engines_env.sh   # overlays + import preflight; edge-ai env untouched
test -d "$CIFAR_DATA/cifar-100-python"
test -f "$STUDY_BASE/split_seed42.json"
test -f "$STAGE_A/predictions.npz" && test -f "$STAGE_A/costs_pipeline.json"
for f in evidence/labels.npy evidence/images.npy evidence/bank.json code/hpc_router_arch_sweep.py; do
    test -e "$SCREEN/$f" || { echo "Missing $SCREEN/$f" >&2; exit 1; }
done
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
out="${ROUTERS_OUT:-${STUDY_BASE}/a40_routers_v1}"
hostname; date -u; nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
git rev-parse HEAD
srun --ntasks=1 "$HPC_PYTHON" -m hpc.a40_routers --base "$STUDY_BASE" --data "$CIFAR_DATA" \
    --screen "$SCREEN" --stage-a "$STAGE_A" --output "$out"
"$HPC_PYTHON" -m hpc.a40_policy --stage-a "$STAGE_A" --scores "$out"/scores/*.npz \
    --output "${STUDY_BASE}/a40_policy_v1"
date -u
